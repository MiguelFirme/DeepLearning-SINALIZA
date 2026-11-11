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
        self.target_frames = settings.target_frames

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self, model_path: str | None = None):
        """Carrega o checkpoint treinado com metadados e pesos estritos."""
        self._loaded = False
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
        if input_dim != LandmarkNormalizer(346, NormalizationConfig()).get_output_dim():
            raise ValueError(f"Checkpoint espera {input_dim} features, mas o pipeline produz 697")

        # Processadores
        self._normalizer = LandmarkNormalizer(346, NormalizationConfig())
        self._seq_processor = SequenceProcessor(SequenceConfig(target_frames=self.target_frames))

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
                predictor: SignPredictor | FusionSessionPredictor | None = None) -> dict:
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
                    "fusion_manifest": settings.fusion_manifest_path}
        if self._predictor:
            return self._predictor.get_model_info()
        return {"status": "not_loaded"}


# Singleton
prediction_service = PredictionService()
