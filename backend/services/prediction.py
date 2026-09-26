"""Serviço de predição — singleton que carrega o modelo uma vez."""
from __future__ import annotations
import logging
import time
from pathlib import Path

import numpy as np

from ml.inference.predictor import SignPredictor, PredictorConfig
from ml.features.normalization import LandmarkNormalizer, NormalizationConfig
from ml.features.sequence import SequenceProcessor, SequenceConfig
from backend.config import settings

logger = logging.getLogger(__name__)


class PredictionService:
    """Serviço singleton para predição em tempo real."""

    def __init__(self):
        self._predictor: SignPredictor | None = None
        self._normalizer: LandmarkNormalizer | None = None
        self._seq_processor: SequenceProcessor | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self, model_path: str | None = None):
        """Carrega modelo do artefato de produção."""
        path = Path(model_path or settings.model_path)
        if not path.exists():
            logger.warning(f"Modelo não encontrado em {path}. API rodará sem modelo.")
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
        self._predictor = SignPredictor(config)

        model_type = artifact.get("model_type", "bilstm")
        num_classes = artifact["num_classes"]
        label_map = artifact.get("label_map", {})

        # Detectar input_dim do state_dict
        state_dict = artifact["model_state_dict"]
        first_weight = next(iter(state_dict.values()))
        input_dim = 0
        for k, v in state_dict.items():
            if "norm.weight" in k or "input_norm.weight" in k:
                input_dim = v.shape[0]
                break
        if input_dim == 0:
            input_dim = 346  # fallback

        self._predictor.load_checkpoint.__wrapped__ = None  # skip file load
        from ml.models import create_model
        model = create_model(model_type, input_dim=input_dim, num_classes=num_classes)
        model.load_state_dict(state_dict, strict=False)
        model.to(config.device)
        model.eval()
        self._predictor.model = model
        self._predictor.num_classes = num_classes
        self._predictor.load_label_names(label_map)

        # Processadores
        self._normalizer = LandmarkNormalizer(346, NormalizationConfig())
        self._seq_processor = SequenceProcessor(SequenceConfig(target_frames=settings.target_frames))

        self._loaded = True
        logger.info(f"Modelo carregado: {model_type} | {num_classes} classes | device={settings.device}")

    def predict(self, landmarks: np.ndarray, mask: np.ndarray | None = None) -> dict:
        """Predição a partir de landmarks brutos."""
        if not self._loaded or self._predictor is None:
            return {"sign": None, "confidence": 0.0, "top_k": [], "is_cooldown": False, "error": "Modelo não carregado"}

        start = time.time()

        # Normalizar
        if self._normalizer:
            # A API antiga aceita máscara temporal (T,). O normalizador usa a
            # máscara de detecção por parte (T, 4), quando ela estiver presente.
            part_mask = mask if mask is not None and mask.ndim == 2 and mask.shape[1] == 4 else None
            landmarks = self._normalizer.normalize_sequence(landmarks, part_mask)

        # Processar sequência
        if self._seq_processor:
            seq, seq_mask = self._seq_processor.process(landmarks)
        else:
            seq, seq_mask = landmarks, None

        result = self._predictor.predict(seq, seq_mask)
        result["processing_time_ms"] = (time.time() - start) * 1000
        return result

    def reset(self):
        if self._predictor:
            self._predictor.reset_state()

    def get_info(self) -> dict:
        if self._predictor:
            return self._predictor.get_model_info()
        return {"status": "not_loaded"}


# Singleton
prediction_service = PredictionService()
