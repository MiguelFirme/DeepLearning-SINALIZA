"""Frozen two-model candidate fusion for the 25 approved signs."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ml.features.feature_modes import (
    FACE16_MEDIAPIPE_INDICES, mask_normalized_face16, select_feature_mode,
)
from ml.features.normalization import LandmarkNormalizer, NormalizationConfig
from ml.features.sequence import SequenceConfig, SequenceProcessor
from ml.inference.predictor import PredictorConfig, SignPredictor
from ml.models import create_model


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_predictor(path: Path, label_map: dict[str, int], device: str) -> tuple[SignPredictor, str, int]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_labels = artifact.get("label_map")
    if checkpoint_labels is None:
        checkpoint_labels = json.loads((path.parent / "data_split" / "label_map.json").read_text(encoding="utf-8"))
    if checkpoint_labels != label_map or artifact.get("num_classes") != 25:
        raise ValueError(f"Checkpoint com vocabulário diferente: {path}")
    mode = artifact.get("feature_mode")
    if mode not in {"pose_face", "full", "pose_face16", "full_face16"}:
        raise ValueError(f"Feature mode inválido: {path}")
    state = artifact["model_state_dict"]
    input_dim = int(state["input_norm.weight"].shape[0])
    if input_dim != LandmarkNormalizer(346, NormalizationConfig()).get_output_dim():
        raise ValueError(f"Dimensão de entrada inválida: {path}")
    model = create_model(artifact["model_type"], input_dim=input_dim,
                         num_classes=25, **artifact.get("model_kwargs", {}))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    predictor = SignPredictor(PredictorConfig(device=device, smoothing_type="none"))
    predictor.model = model
    predictor.num_classes = 25
    predictor.load_label_names(label_map)
    return predictor, mode, int(artifact.get("target_frames", 48))


class FusionBundle:
    def __init__(self, manifest_path: str | Path, device: str):
        path = Path(manifest_path)
        self.spec = json.loads(path.read_text(encoding="utf-8"))
        spec = self.spec
        if spec.get("schema_version") != 1 or spec.get("selected_mode") not in {"fusion", "single_full"}:
            raise ValueError("Manifesto de fusão inválido")
        if "face_indices" in spec and tuple(spec["face_indices"]) != FACE16_MEDIAPIPE_INDICES:
            raise ValueError("Seleção facial diferente do protocolo congelado")
        self.label_path = Path(spec["label_map"])
        paths = {"pose_face": Path(spec["pose_checkpoint"]), "full": Path(spec["full_checkpoint"])}
        for name, file in [("label_map", self.label_path), *paths.items()]:
            expected = spec["label_map_sha256" if name == "label_map" else f"{name if name == 'full' else 'pose'}_sha256"]
            if _digest(file) != expected:
                raise ValueError(f"Hash diferente do protocolo congelado: {file}")
        for name in ("split", "data_manifest", "validation_predictions"):
            # Training/evaluation data is often not distributed with the web
            # app. Verify it when present; only weights/labels are required to
            # serve predictions. All hashes remain frozen in the manifest.
            data_path = Path(spec[f"{name}_path"])
            if data_path.exists() and _digest(data_path) != spec[f"{name}_sha256"]:
                raise ValueError(f"Hash diferente do protocolo congelado: {name}")
        label_map = json.loads(self.label_path.read_text(encoding="utf-8"))
        if len(label_map) != 25 or set(label_map.values()) != set(range(25)):
            raise ValueError("Label map fora do escopo de 25 classes")
        self.labels = {int(index): name for name, index in label_map.items()}
        self.models = {}
        self.feature_modes = {}
        for name, file in paths.items():
            predictor, mode, frames = _load_predictor(file, label_map, device)
            expected_modes = {"pose_face": {"pose_face", "pose_face16"},
                              "full": {"full", "full_face16"}}
            if mode not in expected_modes[name] or frames != spec["target_frames"]:
                raise ValueError(f"Checkpoint incompatível com protocolo: {file}")
            self.models[name] = predictor.model
            self.feature_modes[name] = mode
        self.device = device
        self.target_frames = int(spec["target_frames"])
        self.normalizer = LandmarkNormalizer(346, NormalizationConfig())
        self.sequence = SequenceProcessor(SequenceConfig(target_frames=self.target_frames))

    def probabilities(self, landmarks: np.ndarray, mask: np.ndarray | None) -> tuple[dict, dict]:
        if landmarks.ndim != 2 or landmarks.shape[1] != 346:
            raise ValueError(f"Esperado landmarks (T, 346), recebido {landmarks.shape}")
        probabilities, timings = {}, {}
        for name in ("pose_face", "full"):
            started = time.perf_counter()
            selected, selected_mask = select_feature_mode(landmarks, mask, self.feature_modes[name])
            part_mask = selected_mask if selected_mask is not None and selected_mask.ndim == 2 else None
            normalized = self.normalizer.normalize_sequence(selected, part_mask)
            normalized = mask_normalized_face16(normalized, self.feature_modes[name])
            sequence, seq_mask = self.sequence.process(normalized)
            timings[f"{name}_preprocess_ms"] = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            with torch.inference_mode():
                seq_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
                mask_tensor = torch.from_numpy(seq_mask).unsqueeze(0).to(self.device)
                logits = self.models[name](seq_tensor, mask_tensor)
                logits = logits / float(self.spec["temperatures"][name])
                probabilities[name] = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            timings[f"{name}_inference_ms"] = (time.perf_counter() - started) * 1000
        return probabilities, timings

    def score(self, probabilities: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        full = probabilities["full"]
        if self.spec["selected_mode"] == "single_full":
            return full, np.arange(len(full))
        rule = self.spec["fusion"]
        proposal = probabilities[rule["first"]]
        candidates = np.argsort(-proposal, kind="stable")[:int(rule["top_k"])]
        weight = float(rule["pose_weight"])
        scores = weight * probabilities["pose_face"] + (1 - weight) * full
        # Preserve the full-distribution mass. Renormalizing the top five would
        # inflate confidence and make the 0.30 gate misleading.
        restricted = np.zeros_like(scores)
        restricted[candidates] = scores[candidates]
        return restricted, candidates


class FusionSessionPredictor:
    """Per-websocket state; model weights remain shared."""
    def __init__(self, bundle: FusionBundle, confidence_threshold: float,
                 smoothing_type: str, ema_alpha: float, top_k: int):
        self.bundle = bundle
        self.num_classes = 25
        self.confidence_threshold = confidence_threshold
        self.smoothing_type = smoothing_type
        self.ema_alpha = ema_alpha
        self.top_k = top_k
        self.reset_state()

    def reset_state(self):
        self._ema = None
        self._last_prediction_time = 0.0

    def present(self, probabilities: dict[str, np.ndarray], timings: dict) -> dict:
        now = time.time()
        if now - self._last_prediction_time < 1.0:
            return {"sign": None, "confidence": 0.0, "top_k": [], "is_cooldown": True,
                    "timings": timings}
        scores, candidates = self.bundle.score(probabilities)
        if self.smoothing_type == "ema":
            self._ema = scores.copy() if self._ema is None else self.ema_alpha * scores + (1 - self.ema_alpha) * self._ema
            scores = self._ema.copy()
            if self.bundle.spec["selected_mode"] == "fusion":
                eligible = np.zeros_like(scores)
                eligible[candidates] = scores[candidates]
                scores = eligible
        ranked = np.argsort(-scores, kind="stable")[:self.top_k]
        top = [{"class_id": int(i), "sign": self.bundle.labels[int(i)],
                "confidence": float(scores[i])} for i in ranked if scores[i] > 0]
        best = top[0] if top else None
        confidence = best["confidence"] if best else 0.0
        sign = best["sign"] if best and confidence >= self.confidence_threshold else None
        if sign:
            self._last_prediction_time = now
        return {"sign": sign, "confidence": confidence, "top_k": top,
                "is_cooldown": False, "timings": timings,
                "fusion_mode": self.bundle.spec["selected_mode"]}
