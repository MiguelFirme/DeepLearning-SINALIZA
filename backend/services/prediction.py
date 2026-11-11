"""Serviço de predição — singleton que carrega o modelo uma vez."""
from __future__ import annotations
import logging
import time
import json
from pathlib import Path

import numpy as np

from ml.inference.predictor import SignPredictor, PredictorConfig
from ml.features.normalization import LandmarkNormalizer, NormalizationConfig
from ml.features.sequence import SequenceProcessor, SequenceConfig
from ml.features.feature_modes import FEATURE_MODES, select_feature_mode, mask_normalized_face16
from backend.config import settings
from backend.services.fusion import FusionBundle, FusionSessionPredictor

logger = logging.getLogger(__name__)


class PredictionService:
    """Serviço singleton para predição em tempo real."""

    def __init__(self):
        self._predictor: SignPredictor | None = None
        self._normalizer: LandmarkNormalizer | None = None
        self._seq_processor: SequenceProcessor | None = None
        self._loaded = False
        self._fusion: FusionBundle | None = None
        self.feature_mode = "full"
        self.pipeline_version = 1
        self.pipeline_cfg = None
        self.target_frames = settings.target_frames

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self, model_path: str | None = None):
        """Carrega o checkpoint treinado com metadados e pesos estritos."""
        self._loaded = False
        self.pipeline_version = 1
        self.pipeline_cfg = None
        if model_path is None and settings.fusion_manifest_path:
            self._fusion = FusionBundle(settings.fusion_manifest_path, settings.device)
            self._predictor = None
            self.feature_mode = "fusion" if self._fusion.spec["selected_mode"] == "fusion" else "full"
            self.target_frames = self._fusion.target_frames
            self._loaded = True
            logger.info("Fusão de 25 classes carregada: %s", settings.fusion_manifest_path)
            return
        self._fusion = None
        path = Path(model_path or settings.model_path)
        if not path.exists():
            logger.warning("Modelo não encontrado em %s. API rodará sem modelo.", path)
            return

        import torch
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        self.pipeline_version = int(artifact.get("pipeline_version", 1))
        if self.pipeline_version not in (1, 2):
            raise ValueError("Versão de pipeline não suportada")
        if self.pipeline_version == 2:
            from ml.features.pipeline_v2 import PipelineV2Config, FEATURE_DIM_V2
            self.pipeline_cfg = PipelineV2Config(**artifact["pipeline_config"])
            if (artifact.get("feature_dim") != FEATURE_DIM_V2
                    or artifact.get("target_fps") != self.pipeline_cfg.target_fps
                    or artifact.get("window_seconds") != self.pipeline_cfg.window_seconds
                    or artifact.get("target_frames") != self.pipeline_cfg.n_samples):
                raise ValueError("Metadados v2 inconsistentes com a configuração de features")

        config = PredictorConfig(
            confidence_threshold=settings.confidence_threshold,
            top_k=settings.top_k,
            smoothing_type=settings.smoothing_type,
            ema_alpha=settings.ema_alpha,
            temperature=settings.temperature,
            device=settings.device,
        )
        model_type = artifact.get("model_type")
        if not model_type:
            raise ValueError(f"Checkpoint sem model_type: {path}")
        self._model_type = model_type
        num_classes = artifact["num_classes"]
        label_map = artifact.get("label_map")
        if not label_map:
            label_path = path.parent / "data_split" / "label_map.json"
            if not label_path.exists():
                raise FileNotFoundError(f"Mapa de classes não encontrado: {label_path}")
            label_map = json.loads(label_path.read_text(encoding="utf-8"))
        if len(label_map) != num_classes or set(label_map.values()) != set(range(num_classes)):
            raise ValueError("Mapa de classes incompatível com o checkpoint")

        # Detectar input_dim do state_dict
        state_dict = artifact["model_state_dict"]
        input_norm_weight = state_dict.get("input_norm.weight")
        if input_norm_weight is None:
            raise ValueError("Checkpoint sem input_norm.weight; dimensão de entrada desconhecida")
        input_dim = int(input_norm_weight.shape[0])
        model_kwargs = artifact.get("model_kwargs", {})
        from ml.models import create_model
        model = create_model(model_type, input_dim=input_dim, num_classes=num_classes, **model_kwargs)
        model.load_state_dict(state_dict, strict=True)
        model.to(config.device)
        model.eval()
        predictor = SignPredictor(config)
        predictor.model = model
        predictor.num_classes = num_classes
        predictor.load_label_names(label_map)
        self._predictor = predictor

        self.feature_mode = artifact.get("feature_mode", "full")
        if self.feature_mode not in FEATURE_MODES:
            raise ValueError(f"feature_mode desconhecido: {self.feature_mode}")
        self.target_frames = int(artifact.get("target_frames", settings.target_frames))
        expected_dim = 725 if self.pipeline_version == 2 else LandmarkNormalizer(346, NormalizationConfig()).get_output_dim()
        if input_dim != expected_dim:
            raise ValueError(f"Checkpoint espera {input_dim} features, mas o pipeline produz {expected_dim}")

        # Processadores
        self._normalizer = LandmarkNormalizer(346, NormalizationConfig()) if self.pipeline_version == 1 else None
        self._seq_processor = SequenceProcessor(SequenceConfig(target_frames=self.target_frames)) if self.pipeline_version == 1 else None

        self._loaded = True
        logger.info(f"Modelo carregado: {model_type} | {num_classes} classes | device={settings.device}")

    def new_session_predictor(self) -> SignPredictor | FusionSessionPredictor:
        """Compartilha pesos, com suavização/cooldown separados por conexão."""
        if self._fusion is not None:
            return FusionSessionPredictor(self._fusion, settings.confidence_threshold,
                                          settings.smoothing_type, settings.ema_alpha, settings.top_k)
        if self._predictor is None:
            raise RuntimeError("Modelo não carregado")
        predictor = SignPredictor(self._predictor.config)
        predictor.model = self._predictor.model
        predictor.num_classes = self._predictor.num_classes
        predictor.label_names = self._predictor.label_names
        return predictor

    def predict(self, landmarks: np.ndarray, mask: np.ndarray | None = None,
                predictor: SignPredictor | FusionSessionPredictor | None = None,
                timestamps=None, width=None, height=None) -> dict:
        """Predição a partir de landmarks brutos."""
        if self._loaded and self._fusion is not None:
            start = time.perf_counter()
            probabilities, timings = self._fusion.probabilities(landmarks, mask)
            session = predictor or self.new_session_predictor()
            result = session.present(probabilities, timings)
            result["processing_time_ms"] = (time.perf_counter() - start) * 1000
            return result
        if not self._loaded or self._predictor is None:
            return {"sign": None, "confidence": 0.0, "top_k": [], "is_cooldown": False, "error": "Modelo não carregado"}

        if self.pipeline_version == 2:
            from ml.features.pipeline_v2 import prepare_features_v2
            if timestamps is None or mask is None or width is None or height is None:
                raise ValueError("V2 exige timestamps, máscara por parte, largura e altura")
            start = time.perf_counter()
            seq, seq_mask, _ = prepare_features_v2(landmarks, mask, timestamps, width, height, self.pipeline_cfg)
            session = predictor or self._predictor
            result = session.predict(seq, seq_mask)
            probabilities = result.get("raw_probs")
            if probabilities is not None and session.label_names[int(np.argmax(probabilities))] == "_nenhum_sinal":
                result.update(status="no_sign", sign=None, top_k=[])
                session._last_prediction_time = 0.
            result["processing_time_ms"] = (time.perf_counter() - start) * 1000
            return result

        start = time.time()

        if landmarks.ndim != 2 or landmarks.shape[1] != 346:
            raise ValueError(f"Esperado landmarks (T, 346), recebido {landmarks.shape}")
        landmarks, mask = select_feature_mode(landmarks, mask, self.feature_mode)

        # Normalizar
        if self._normalizer:
            # A API antiga aceita máscara temporal (T,). O normalizador usa a
            # máscara de detecção por parte (T, 4), quando ela estiver presente.
            part_mask = mask if mask is not None and mask.ndim == 2 and mask.shape[1] == 4 else None
            landmarks = self._normalizer.normalize_sequence(landmarks, part_mask)
            landmarks = mask_normalized_face16(landmarks, self.feature_mode)

        # Processar sequência
        if self._seq_processor:
            seq, seq_mask = self._seq_processor.process(landmarks)
        else:
            seq, seq_mask = landmarks, None

        result = (predictor or self._predictor).predict(seq, seq_mask)
        result["processing_time_ms"] = (time.time() - start) * 1000
        return result

    def reset(self):
        if self._predictor:
            self._predictor.reset_state()

    def get_info(self) -> dict:
        if self._fusion is not None:
            model_types = set(self._fusion.model_types.values())
            model_type = f"{next(iter(model_types))}_fusion" if len(model_types) == 1 else "mixed_model_fusion"
            return {"status": "loaded", "num_classes": 25,
                    "type": model_type, "device": self._fusion.device,
                    "trainable": sum(p.numel() for model in self._fusion.models.values()
                                     for p in model.parameters() if p.requires_grad),
                    "smoothing": settings.smoothing_type,
                    "feature_mode": self.feature_mode,
                    "feature_modes": self._fusion.feature_modes,
                    "model_types": self._fusion.model_types,
                    "pipeline_version": 1,
                    "labels": [self._fusion.labels[i] for i in sorted(self._fusion.labels)],
                    "fusion_manifest": settings.fusion_manifest_path}
        if self._predictor:
            info = self._predictor.get_model_info()
            info["pipeline_version"] = self.pipeline_version
            info["labels"] = [self._predictor.label_names[i] for i in sorted(self._predictor.label_names)]
            if self.pipeline_version == 2:
                info.update(type=f"{self._model_type}_v2", target_fps=self.pipeline_cfg.target_fps,
                            window_seconds=self.pipeline_cfg.window_seconds, feature_dim=725)
            return info
        return {"status": "not_loaded"}


# Singleton
prediction_service = PredictionService()
