"""Normalização espacial de landmarks com suporte à máscara de detecção."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PART_SLICES = ((0, 63), (63, 126), (126, 226), (226, 346))


@dataclass
class NormalizationConfig:
    center_by_shoulders: bool = True
    normalize_by_shoulder_distance: bool = True
    relative_hand_coords: bool = False
    add_velocity: bool = True
    add_acceleration: bool = False
    add_distances: bool = True
    epsilon: float = 1e-8
    left_shoulder_pose_idx: int = 5
    right_shoulder_pose_idx: int = 6


class LandmarkNormalizer:
    """Normaliza `(T, D)` sem transformar landmarks ausentes em pontos falsos."""

    def __init__(
        self,
        feature_dim: int | NormalizationConfig = 346,
        config: NormalizationConfig | None = None,
    ):
        if isinstance(feature_dim, NormalizationConfig):
            config = feature_dim
            feature_dim = 346
        self.feature_dim = int(feature_dim)
        self.cfg = config or NormalizationConfig()

    def normalize_sequence(
        self,
        seq: np.ndarray,
        part_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Normaliza coordenadas e acrescenta features temporais e distâncias."""
        if seq.ndim != 2:
            raise ValueError(f"Esperado landmarks (T, D), recebido {seq.shape}")
        if seq.shape[0] == 0:
            return np.empty((0, self.get_output_dim()), dtype=np.float32)

        raw = np.nan_to_num(
            seq.copy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        mask = self._coerce_part_mask(raw, part_mask)
        shoulder_center, shoulder_scale = self._get_shoulder_ref(raw, mask)

        if shoulder_center is not None:
            self._normalize_xyz_regions(raw, shoulder_center, shoulder_scale)
        self._zero_missing_parts(raw, mask)

        parts = [raw]
        if self.cfg.add_velocity:
            parts.append(self._compute_velocity(raw, mask))
        if self.cfg.add_acceleration:
            parts.append(self._compute_acceleration(raw, mask))
        if self.cfg.add_distances:
            parts.append(self._compute_distances(raw, mask))

        result = np.concatenate(parts, axis=-1)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def normalize(self, seq: np.ndarray, part_mask: np.ndarray | None = None) -> np.ndarray:
        """Alias compatível com integrações anteriores."""
        return self.normalize_sequence(seq, part_mask)

    def get_output_dim(self) -> int:
        dim = self.feature_dim
        if self.cfg.add_velocity:
            dim += self.feature_dim
        if self.cfg.add_acceleration:
            dim += self.feature_dim
        if self.cfg.add_distances:
            dim += 5
        return dim

    def _coerce_part_mask(self, seq: np.ndarray, part_mask: np.ndarray | None) -> np.ndarray:
        expected = (seq.shape[0], 4)
        if part_mask is not None:
            mask = np.asarray(part_mask, dtype=bool)
            if mask.shape != expected:
                raise ValueError(f"Máscara deve ter shape {expected}, recebido {mask.shape}")
            return mask.copy()

        inferred = np.zeros(expected, dtype=bool)
        for part, (start, end) in enumerate(PART_SLICES):
            if start < seq.shape[1]:
                inferred[:, part] = np.any(
                    np.abs(seq[:, start:min(end, seq.shape[1])]) > self.cfg.epsilon,
                    axis=1,
                )
        return inferred

    def _get_shoulder_ref(self, seq: np.ndarray, mask: np.ndarray):
        pose_start = 126
        ls_offset = pose_start + self.cfg.left_shoulder_pose_idx * 4
        rs_offset = pose_start + self.cfg.right_shoulder_pose_idx * 4
        if rs_offset + 3 > seq.shape[1]:
            return None, None

        left = seq[:, ls_offset:ls_offset + 3]
        right = seq[:, rs_offset:rs_offset + 3]
        center = (left + right) / 2.0
        distance = np.linalg.norm(left - right, axis=-1)
        valid = mask[:, 2] & np.isfinite(distance) & (distance > self.cfg.epsilon)
        if not np.any(valid):
            return None, None

        median_center = np.median(center[valid], axis=0)
        median_scale = float(np.median(distance[valid]))
        center[~valid] = median_center
        distance[~valid] = median_scale
        return center.astype(np.float32), distance.astype(np.float32)

    def _normalize_xyz_regions(self, seq, center, scale):
        safe_scale = np.maximum(scale, self.cfg.epsilon)[:, None]
        regions = (
            (0, min(126, seq.shape[1]), 3),
            (126, min(226, seq.shape[1]), 4),
            (226, min(346, seq.shape[1]), 3),
        )
        for start, end, stride in regions:
            for offset in range(start, end, stride):
                xyz = seq[:, offset:offset + 3]
                if self.cfg.center_by_shoulders:
                    xyz -= center
                if self.cfg.normalize_by_shoulder_distance:
                    xyz /= safe_scale

    @staticmethod
    def _zero_missing_parts(seq: np.ndarray, mask: np.ndarray):
        for part, (start, end) in enumerate(PART_SLICES):
            if start < seq.shape[1]:
                seq[~mask[:, part], start:min(end, seq.shape[1])] = 0.0

    def _compute_velocity(self, seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
        velocity = np.zeros_like(seq)
        velocity[1:] = seq[1:] - seq[:-1]
        for part, (start, end) in enumerate(PART_SLICES):
            if start >= seq.shape[1]:
                continue
            valid_pair = mask[1:, part] & mask[:-1, part]
            velocity[1:, start:min(end, seq.shape[1])][~valid_pair] = 0.0
        velocity[:, 129:min(226, seq.shape[1]):4] = 0.0
        return velocity

    def _compute_acceleration(self, seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
        velocity = self._compute_velocity(seq, mask)
        acceleration = np.zeros_like(seq)
        acceleration[2:] = velocity[2:] - velocity[1:-1]
        for part, (start, end) in enumerate(PART_SLICES):
            if start >= seq.shape[1]:
                continue
            valid_triplet = mask[2:, part] & mask[1:-1, part] & mask[:-2, part]
            acceleration[2:, start:min(end, seq.shape[1])][~valid_triplet] = 0.0
        acceleration[:, 129:min(226, seq.shape[1]):4] = 0.0
        return acceleration

    def _compute_distances(self, seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
        distances = np.zeros((seq.shape[0], 5), dtype=np.float32)
        if seq.shape[1] < 226:
            return distances

        left_hand = seq[:, :63].reshape(-1, 21, 3).mean(axis=1)
        right_hand = seq[:, 63:126].reshape(-1, 21, 3).mean(axis=1)
        nose = seq[:, 126:129]
        trunk = np.zeros_like(nose)

        both_hands = mask[:, 0] & mask[:, 1]
        left_pose = mask[:, 0] & mask[:, 2]
        right_pose = mask[:, 1] & mask[:, 2]
        distances[both_hands, 0] = np.linalg.norm(
            left_hand[both_hands] - right_hand[both_hands], axis=-1
        )
        distances[left_pose, 1] = np.linalg.norm(
            left_hand[left_pose] - nose[left_pose], axis=-1
        )
        distances[right_pose, 2] = np.linalg.norm(
            right_hand[right_pose] - nose[right_pose], axis=-1
        )
        distances[left_pose, 3] = np.linalg.norm(
            left_hand[left_pose] - trunk[left_pose], axis=-1
        )
        distances[right_pose, 4] = np.linalg.norm(
            right_hand[right_pose] - trunk[right_pose], axis=-1
        )
        return distances
