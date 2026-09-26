"""Aumentação coerente para sequências de landmarks brutos."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PART_SLICES = ((0, 63), (63, 126), (126, 226), (226, 346))
COORD_REGIONS = (
    (0, 63, 3, 0),
    (63, 126, 3, 1),
    (126, 226, 4, 2),
    (226, 346, 3, 3),
)


@dataclass
class AugmentationConfig:
    enabled: bool = True
    prob: float = 0.5
    noise_enabled: bool = True
    noise_prob: float = 0.5
    noise_std: float = 0.005
    scale_enabled: bool = True
    scale_prob: float = 0.5
    scale_range: tuple[float, float] = (0.9, 1.1)
    translate_enabled: bool = True
    translate_prob: float = 0.5
    translate_range: float = 0.03
    rotate_enabled: bool = True
    rotate_prob: float = 0.5
    rotate_range_deg: float = 10.0
    temporal_crop_enabled: bool = True
    temporal_crop_prob: float = 0.5
    crop_ratio_range: tuple[float, float] = (0.8, 1.0)
    frame_drop_enabled: bool = True
    frame_drop_apply_prob: float = 0.5
    frame_drop_prob: float = 0.05
    speed_enabled: bool = True
    speed_prob: float = 0.5
    speed_range: tuple[float, float] = (0.9, 1.1)
    landmark_mask_enabled: bool = True
    landmark_mask_apply_prob: float = 0.5
    mask_prob: float = 0.02
    temporal_jitter_enabled: bool = True
    temporal_jitter_prob: float = 0.5
    jitter_range: int = 1
    mirror_enabled: bool = False

    @classmethod
    def from_mapping(cls, values: dict | None) -> "AugmentationConfig":
        """Cria configuração aceitando o formato existente em `default.yaml`."""
        values = values or {}
        config = cls(
            enabled=bool(values.get("enabled", True)),
            prob=float(values.get("prob", 1.0)),
        )

        noise = values.get("gaussian_noise", {})
        config.noise_prob = float(noise.get("prob", config.noise_prob))
        config.noise_enabled = config.noise_prob > 0
        config.noise_std = float(noise.get("std", config.noise_std))

        scale = values.get("scale_jitter", {})
        config.scale_prob = float(scale.get("prob", config.scale_prob))
        config.scale_enabled = config.scale_prob > 0
        config.scale_range = (
            float(scale.get("min_scale", config.scale_range[0])),
            float(scale.get("max_scale", config.scale_range[1])),
        )

        translation = values.get("translation_jitter", {})
        config.translate_prob = float(translation.get("prob", 0.0))
        config.translate_enabled = config.translate_prob > 0
        config.translate_range = float(
            translation.get("max_offset", config.translate_range)
        )

        rotation = values.get("rotation_jitter", {})
        config.rotate_prob = float(rotation.get("prob", config.rotate_prob))
        config.rotate_enabled = config.rotate_prob > 0
        config.rotate_range_deg = float(rotation.get("max_angle", config.rotate_range_deg))

        crop = values.get("temporal_crop", {})
        config.temporal_crop_prob = float(crop.get("prob", config.temporal_crop_prob))
        config.temporal_crop_enabled = config.temporal_crop_prob > 0
        config.crop_ratio_range = (
            float(crop.get("min_ratio", config.crop_ratio_range[0])),
            1.0,
        )

        speed = values.get("speed_warp", {})
        config.speed_prob = float(speed.get("prob", config.speed_prob))
        config.speed_enabled = config.speed_prob > 0
        config.speed_range = (
            float(speed.get("min_rate", config.speed_range[0])),
            float(speed.get("max_rate", config.speed_range[1])),
        )

        frame_drop = values.get("frame_drop", {})
        config.frame_drop_apply_prob = float(
            frame_drop.get("prob", config.frame_drop_apply_prob)
        )
        config.frame_drop_enabled = config.frame_drop_apply_prob > 0
        config.frame_drop_prob = float(
            frame_drop.get("max_drop_ratio", config.frame_drop_prob)
        )

        landmark_mask = values.get("landmark_mask", {})
        config.landmark_mask_apply_prob = float(landmark_mask.get("prob", 0.0))
        config.landmark_mask_enabled = config.landmark_mask_apply_prob > 0
        config.mask_prob = float(landmark_mask.get("mask_prob", config.mask_prob))

        temporal_shift = values.get("temporal_shift", {})
        config.temporal_jitter_prob = float(
            temporal_shift.get("prob", config.temporal_jitter_prob)
        )
        config.temporal_jitter_enabled = config.temporal_jitter_prob > 0
        config.jitter_range = int(
            temporal_shift.get("max_shift", config.jitter_range)
        )
        return config


class SequenceAugmentor:
    """Aplica transformações nos landmarks e mantém a máscara sincronizada."""

    def __init__(self, config: AugmentationConfig | None = None, rng=None):
        self.config = config or AugmentationConfig()
        self.rng = rng or np.random.default_rng()

    def augment(self, seq: np.ndarray) -> np.ndarray:
        """API compatível: aumenta somente a sequência e devolve um array."""
        result, _ = self.augment_with_mask(seq, None)
        return result

    def augment_with_mask(
        self,
        seq: np.ndarray,
        part_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        result = seq.copy().astype(np.float32)
        mask = self._coerce_mask(result, part_mask)
        if not self.config.enabled or self.rng.random() > self.config.prob:
            self._zero_missing_parts(result, mask)
            return result, mask

        cfg = self.config
        if cfg.noise_enabled and self._coin(cfg.noise_prob):
            result = self._add_noise(result, mask)
        if cfg.scale_enabled and self._coin(cfg.scale_prob):
            result = self._scale(result, mask)
        if cfg.translate_enabled and self._coin(cfg.translate_prob):
            result = self._translate(result, mask)
        if cfg.rotate_enabled and self._coin(cfg.rotate_prob):
            result = self._rotate_2d(result, mask)
        if cfg.temporal_crop_enabled and self._coin(cfg.temporal_crop_prob):
            result, mask = self._temporal_crop(result, mask)
        if cfg.frame_drop_enabled and self._coin(cfg.frame_drop_apply_prob):
            result, mask = self._frame_drop(result, mask)
        if cfg.speed_enabled and self._coin(cfg.speed_prob):
            result, mask = self._speed_variation(result, mask)
        if cfg.landmark_mask_enabled and self._coin(cfg.landmark_mask_apply_prob):
            result = self._landmark_mask(result, mask)
        if cfg.temporal_jitter_enabled and self._coin(cfg.temporal_jitter_prob):
            result, mask = self._temporal_jitter(result, mask)

        self._zero_missing_parts(result, mask)
        return result.astype(np.float32), mask.astype(bool)

    def _coin(self, probability: float) -> bool:
        return self.rng.random() < probability

    @staticmethod
    def _coerce_mask(seq, part_mask):
        if part_mask is not None:
            mask = np.asarray(part_mask, dtype=bool)
            if mask.shape != (seq.shape[0], 4):
                raise ValueError(
                    f"Máscara deve ter shape {(seq.shape[0], 4)}, recebido {mask.shape}"
                )
            return mask.copy()
        mask = np.zeros((seq.shape[0], 4), dtype=bool)
        for part, (start, end) in enumerate(PART_SLICES):
            if start < seq.shape[1]:
                mask[:, part] = np.any(np.abs(seq[:, start:min(end, seq.shape[1])]) > 1e-8, axis=1)
        return mask

    @staticmethod
    def _valid_rows(mask, part):
        return mask[:, part]

    def _coordinate_indices(self, dim):
        for start, end, stride, part in COORD_REGIONS:
            end = min(end, dim)
            if start >= end:
                continue
            bases = np.arange(start, end, stride)
            yield part, bases, bases + 1, bases + 2

    def _add_noise(self, seq, mask):
        result = seq.copy()
        for part, ix, iy, iz in self._coordinate_indices(seq.shape[1]):
            rows = self._valid_rows(mask, part)
            indices = np.concatenate([ix, iy, iz])
            noise = self.rng.normal(0, self.config.noise_std, (rows.sum(), len(indices)))
            result[np.ix_(rows, indices)] += noise.astype(np.float32)
        return result

    def _scale(self, seq, mask):
        result = seq.copy()
        factor = self.rng.uniform(*self.config.scale_range)
        for part, ix, iy, iz in self._coordinate_indices(seq.shape[1]):
            rows = self._valid_rows(mask, part)
            result[np.ix_(rows, np.concatenate([ix, iy, iz]))] *= factor
        return result

    def _translate(self, seq, mask):
        result = seq.copy()
        dx, dy = self.rng.uniform(
            -self.config.translate_range, self.config.translate_range, size=2
        )
        for part, ix, iy, _ in self._coordinate_indices(seq.shape[1]):
            rows = self._valid_rows(mask, part)
            result[np.ix_(rows, ix)] += dx
            result[np.ix_(rows, iy)] += dy
        return result

    def _rotate_2d(self, seq, mask):
        result = seq.copy()
        angle = np.deg2rad(
            self.rng.uniform(-self.config.rotate_range_deg, self.config.rotate_range_deg)
        )
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for part, ix, iy, _ in self._coordinate_indices(seq.shape[1]):
            rows = self._valid_rows(mask, part)
            x = result[np.ix_(rows, ix)].copy()
            y = result[np.ix_(rows, iy)].copy()
            result[np.ix_(rows, ix)] = x * cos_a - y * sin_a
            result[np.ix_(rows, iy)] = x * sin_a + y * cos_a
        return result

    def _temporal_crop(self, seq, mask):
        if seq.shape[0] < 4:
            return seq, mask
        ratio = self.rng.uniform(*self.config.crop_ratio_range)
        new_length = max(2, int(seq.shape[0] * ratio))
        start = self.rng.integers(0, max(1, seq.shape[0] - new_length + 1))
        return seq[start:start + new_length], mask[start:start + new_length]

    def _frame_drop(self, seq, mask):
        if seq.shape[0] < 4:
            return seq, mask
        keep = self.rng.random(seq.shape[0]) > self.config.frame_drop_prob
        keep[[0, -1]] = True
        return seq[keep], mask[keep]

    def _speed_variation(self, seq, mask):
        if seq.shape[0] < 4:
            return seq, mask
        new_length = max(2, int(seq.shape[0] * self.rng.uniform(*self.config.speed_range)))
        indices = np.linspace(0, seq.shape[0] - 1, new_length).astype(int)
        return seq[indices], mask[indices]

    def _landmark_mask(self, seq, mask):
        result = seq.copy()
        for part, ix, iy, iz in self._coordinate_indices(seq.shape[1]):
            rows = self._valid_rows(mask, part)
            drop = self.rng.random((rows.sum(), len(ix))) < self.config.mask_prob
            for indices in (ix, iy, iz):
                values = result[np.ix_(rows, indices)]
                values[drop] = 0.0
                result[np.ix_(rows, indices)] = values
        return result

    def _temporal_jitter(self, seq, mask):
        if seq.shape[0] < 3:
            return seq, mask
        offsets = self.rng.integers(
            -self.config.jitter_range, self.config.jitter_range + 1, size=seq.shape[0]
        )
        indices = np.clip(np.arange(seq.shape[0]) + offsets, 0, seq.shape[0] - 1)
        return seq[indices], mask[indices]

    @staticmethod
    def _zero_missing_parts(seq, mask):
        for part, (start, end) in enumerate(PART_SLICES):
            if start < seq.shape[1]:
                seq[~mask[:, part], start:min(end, seq.shape[1])] = 0.0
