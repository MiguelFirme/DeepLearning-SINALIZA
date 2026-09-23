"""
Inferência em tempo real para sinais de Libras.

Features:
    - Carregamento de checkpoint
    - Suavização temporal (EMA ou votação por maioria)
    - Cooldown entre predições
    - Top-K com confiança
    - Temperature scaling
"""
from __future__ import annotations
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models import create_model

logger = logging.getLogger(__name__)


@dataclass
class PredictorConfig:
    confidence_threshold: float = 0.3
    top_k: int = 5
    smoothing_type: str = "ema"  # ema | majority | none
    ema_alpha: float = 0.6
    majority_window: int = 5
    cooldown_seconds: float = 1.0
    temperature: float = 1.0
    device: str = "cpu"


class SignPredictor:
    """Predição em tempo real com suavização temporal."""

    def __init__(self, config: PredictorConfig | None = None):
        self.config = config or PredictorConfig()
        self.device = torch.device(self.config.device)
        self.model: nn.Module | None = None
        self.label_names: dict[int, str] = {}
        self.num_classes: int = 0

        # Estado temporal
        self._ema_probs: np.ndarray | None = None
        self._history: deque = deque(maxlen=self.config.majority_window)
        self._last_prediction_time: float = 0.0

    def load_checkpoint(self, checkpoint_path: str | Path, model_type: str | None = None, **model_kwargs):
        """Carrega modelo de um checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.num_classes = ckpt.get("num_classes", model_kwargs.get("num_classes", 0))
        model_class = model_type or ckpt.get("model_type", ckpt.get("model_class", "bilstm")).lower()

        # Reconstruir modelo
        final_kwargs = {**model_kwargs}
        final_kwargs.setdefault("num_classes", self.num_classes)
        self.model = create_model(model_class, **final_kwargs)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Modelo carregado: {model_class} | {self.num_classes} classes | device={self.device}")

    def load_label_names(self, label_map: dict[str, int]):
        """Carrega mapa de nomes das classes."""
        self.label_names = {v: k for k, v in label_map.items()}

    @torch.no_grad()
    def predict(self, sequence: np.ndarray, mask: np.ndarray | None = None) -> dict:
        """
        Predição para uma sequência.

        Args:
            sequence: (T, D) float32
            mask: (T,) bool, opcional

        Returns:
            dict com sign, confidence, top_k, is_cooldown, raw_probs
        """
        if self.model is None:
            raise RuntimeError("Modelo não carregado. Use load_checkpoint().")

        # Cooldown check
        now = time.time()
        if now - self._last_prediction_time < self.config.cooldown_seconds:
            return {"sign": None, "confidence": 0.0, "top_k": [], "is_cooldown": True}

        # Preparar tensor
        seq_t = torch.from_numpy(sequence).unsqueeze(0).to(self.device)  # (1, T, D)
        mask_t = None
        if mask is not None:
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)  # (1, T)

        # Forward
        logits = self.model(seq_t, mask_t)  # (1, C)

        # Temperature scaling
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature

        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]  # (C,)

        # Suavização temporal
        smoothed = self._apply_smoothing(probs)

        # Top-K
        top_k_idx = np.argsort(smoothed)[-self.config.top_k:][::-1]
        top_k = []
        for idx in top_k_idx:
            name = self.label_names.get(idx, str(idx))
            top_k.append({"class_id": int(idx), "sign": name, "confidence": float(smoothed[idx])})

        # Predição final
        best_idx = top_k_idx[0]
        best_conf = float(smoothed[best_idx])
        best_name = self.label_names.get(best_idx, str(best_idx))

        if best_conf < self.config.confidence_threshold:
            return {"sign": None, "confidence": best_conf, "top_k": top_k, "is_cooldown": False, "raw_probs": smoothed}

        self._last_prediction_time = now
        return {"sign": best_name, "confidence": best_conf, "top_k": top_k, "is_cooldown": False, "raw_probs": smoothed}

    def _apply_smoothing(self, probs: np.ndarray) -> np.ndarray:
        if self.config.smoothing_type == "ema":
            if self._ema_probs is None:
                self._ema_probs = probs.copy()
            else:
                alpha = self.config.ema_alpha
                self._ema_probs = alpha * probs + (1 - alpha) * self._ema_probs
            return self._ema_probs

        elif self.config.smoothing_type == "majority":
            pred = probs.argmax()
            self._history.append(pred)
            if len(self._history) >= self.config.majority_window:
                counts = np.bincount(list(self._history), minlength=len(probs))
                majority_pred = counts.argmax()
                # Retornar probs com peso extra para o voto da maioria
                result = probs.copy()
                result[majority_pred] += 0.2
                result /= result.sum()
                return result
            return probs
        else:
            return probs

    def reset_state(self):
        """Reseta estado temporal (entre vídeos/sessões)."""
        self._ema_probs = None
        self._history.clear()
        self._last_prediction_time = 0.0

    def get_model_info(self) -> dict:
        if self.model is None:
            return {"status": "not_loaded"}
        info = self.model.get_model_info() if hasattr(self.model, "get_model_info") else {}
        info["device"] = str(self.device)
        info["num_classes"] = self.num_classes
        info["smoothing"] = self.config.smoothing_type
        return info
