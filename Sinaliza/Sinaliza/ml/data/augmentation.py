"""
Aumento de dados para sequências de landmarks.

IMPORTANTE: NÃO usar espelhamento horizontal por padrão.
Lateralidade é semântica em Libras — trocar esquerda/direita pode
alterar o significado do sinal.

Técnicas implementadas:
    1. Ruído gaussiano
    2. Escala (zoom espacial)
    3. Translação espacial
    4. Rotação 2D
    5. Recorte temporal (crop + resize)
    6. Frame drop aleatório
    7. Variação de velocidade
    8. Mascaramento de landmarks
    9. Jitter temporal (embaralhar frames próximos)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class AugmentationConfig:
    enabled: bool = True
    prob: float = 0.5  # probabilidade global por amostra

    # Ruído gaussiano
    noise_enabled: bool = True
    noise_std: float = 0.005

    # Escala espacial
    scale_enabled: bool = True
    scale_range: tuple[float, float] = (0.9, 1.1)

    # Translação espacial
    translate_enabled: bool = True
    translate_range: float = 0.03

    # Rotação 2D (xy)
    rotate_enabled: bool = True
    rotate_range_deg: float = 10.0

    # Recorte temporal
    temporal_crop_enabled: bool = True
    crop_ratio_range: tuple[float, float] = (0.8, 1.0)

    # Frame drop
    frame_drop_enabled: bool = True
    frame_drop_prob: float = 0.05

    # Variação de velocidade
    speed_enabled: bool = True
    speed_range: tuple[float, float] = (0.9, 1.1)

    # Mascaramento de landmarks
    landmark_mask_enabled: bool = True
    mask_prob: float = 0.02

    # Jitter temporal
    temporal_jitter_enabled: bool = True
    jitter_range: int = 1

    # Espelhamento horizontal (DESABILITADO por padrão)
    mirror_enabled: bool = False


class SequenceAugmentor:
    """Aplica aumentação estocástica em sequências de landmarks."""

    def __init__(self, config: AugmentationConfig | None = None, rng: np.random.Generator | None = None):
        self.config = config or AugmentationConfig()
        self.rng = rng or np.random.default_rng()

    def augment(self, seq: np.ndarray) -> np.ndarray:
        """
        Aplica aumentações sobre (T, D).
        Retorna cópia modificada; nunca altera o original.
        """
        if not self.config.enabled:
            return seq
        if self.rng.random() > self.config.prob:
            return seq

        seq = seq.copy().astype(np.float32)
        cfg = self.config

        if cfg.noise_enabled and self._coin():
            seq = self._add_noise(seq)
        if cfg.scale_enabled and self._coin():
            seq = self._scale(seq)
        if cfg.translate_enabled and self._coin():
            seq = self._translate(seq)
        if cfg.rotate_enabled and self._coin():
            seq = self._rotate_2d(seq)
        if cfg.temporal_crop_enabled and self._coin():
            seq = self._temporal_crop(seq)
        if cfg.frame_drop_enabled and self._coin():
            seq = self._frame_drop(seq)
        if cfg.speed_enabled and self._coin():
            seq = self._speed_variation(seq)
        if cfg.landmark_mask_enabled and self._coin():
            seq = self._landmark_mask(seq)
        if cfg.temporal_jitter_enabled and self._coin():
            seq = self._temporal_jitter(seq)

        return seq

    # ------ Métodos privados ------

    def _coin(self) -> bool:
        return self.rng.random() < 0.5

    def _add_noise(self, seq: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0, self.config.noise_std, size=seq.shape).astype(np.float32)
        return seq + noise

    def _scale(self, seq: np.ndarray) -> np.ndarray:
        lo, hi = self.config.scale_range
        factor = self.rng.uniform(lo, hi)
        return seq * factor

    def _translate(self, seq: np.ndarray) -> np.ndarray:
        T, D = seq.shape
        offset = self.rng.uniform(-self.config.translate_range, self.config.translate_range, size=(1, D))
        return seq + offset.astype(np.float32)

    def _rotate_2d(self, seq: np.ndarray) -> np.ndarray:
        """Rota coordenadas (x, y) de cada grupo de 3."""
        angle = np.deg2rad(self.rng.uniform(-self.config.rotate_range_deg, self.config.rotate_range_deg))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        result = seq.copy()
        T, D = seq.shape
        # Rotacionar regiões com coordenadas (x,y,z) — hands + face (stride 3)
        for i in range(0, min(D, 126), 3):
            x, y = result[:, i], result[:, i + 1]
            result[:, i] = x * cos_a - y * sin_a
            result[:, i + 1] = x * sin_a + y * cos_a
        # Pose (stride 4: x,y,z,vis)
        for i in range(126, min(D, 226), 4):
            x, y = result[:, i], result[:, i + 1]
            result[:, i] = x * cos_a - y * sin_a
            result[:, i + 1] = x * sin_a + y * cos_a
        # Face (stride 3)
        for i in range(226, D, 3):
            x, y = result[:, i], result[:, i + 1]
            result[:, i] = x * cos_a - y * sin_a
            result[:, i + 1] = x * sin_a + y * cos_a
        return result

    def _temporal_crop(self, seq: np.ndarray) -> np.ndarray:
        T = seq.shape[0]
        if T < 4:
            return seq
        lo, hi = self.config.crop_ratio_range
        ratio = self.rng.uniform(lo, hi)
        new_T = max(2, int(T * ratio))
        start = self.rng.integers(0, max(1, T - new_T + 1))
        return seq[start : start + new_T]

    def _frame_drop(self, seq: np.ndarray) -> np.ndarray:
        T = seq.shape[0]
        if T < 4:
            return seq
        keep = self.rng.random(T) > self.config.frame_drop_prob
        keep[0] = True  # manter primeiro
        keep[-1] = True  # e último
        result = seq[keep]
        return result if result.shape[0] >= 2 else seq

    def _speed_variation(self, seq: np.ndarray) -> np.ndarray:
        T = seq.shape[0]
        if T < 4:
            return seq
        lo, hi = self.config.speed_range
        factor = self.rng.uniform(lo, hi)
        new_T = max(2, int(T * factor))
        idx = np.linspace(0, T - 1, new_T).astype(int)
        return seq[idx]

    def _landmark_mask(self, seq: np.ndarray) -> np.ndarray:
        mask = self.rng.random(seq.shape) < self.config.mask_prob
        result = seq.copy()
        result[mask] = 0.0
        return result

    def _temporal_jitter(self, seq: np.ndarray) -> np.ndarray:
        T = seq.shape[0]
        if T < 3:
            return seq
        r = self.config.jitter_range
        indices = np.arange(T)
        offsets = self.rng.integers(-r, r + 1, size=T)
        new_idx = np.clip(indices + offsets, 0, T - 1)
        return seq[new_idx]
