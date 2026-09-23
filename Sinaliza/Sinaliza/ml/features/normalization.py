"""Normalização espacial robusta de landmarks.

Invariância a: distância da câmera, posição, tamanho corporal, resolução.
Features derivadas: velocidade, aceleração, distâncias entre pontos-chave.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class NormalizationConfig:
    center_by_shoulders: bool = True
    normalize_by_shoulder_distance: bool = True
    relative_hand_coords: bool = True
    add_velocity: bool = True
    add_acceleration: bool = False
    add_distances: bool = True
    epsilon: float = 1e-8
    left_shoulder_pose_idx: int = 5   # index in POSE_SELECTED
    right_shoulder_pose_idx: int = 6
    left_wrist_pose_idx: int = 9
    right_wrist_pose_idx: int = 10

class LandmarkNormalizer:
    """Normaliza sequências de landmarks extraídos."""

    def __init__(self, feature_dim: int, config: NormalizationConfig | None = None):
        self.feature_dim = feature_dim
        self.cfg = config or NormalizationConfig()

    def normalize_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Normaliza sequência (T, D) com centralização, escala e features derivadas."""
        if seq.ndim != 2 or seq.shape[0] == 0:
            return seq

        seq = self._forward_fill(seq.copy().astype(np.float32))

        # Encontrar referência dos ombros (pose starts at offset 126 for hands+hands)
        # Simplificação: normalizar coordenadas globais
        shoulder_center, shoulder_scale = self._get_shoulder_ref(seq)

        if self.cfg.center_by_shoulders and shoulder_center is not None:
            # Subtrair centro (aplicar em blocos de 3 coords)
            for i in range(0, min(seq.shape[1], 126), 3):  # hands region
                seq[:, i:i+3] -= shoulder_center
            # pose region (groups of 4: x,y,z,vis)
            pose_start = 126
            for i in range(pose_start, min(seq.shape[1], pose_start + 100), 4):
                seq[:, i:i+3] -= shoulder_center

        if self.cfg.normalize_by_shoulder_distance and shoulder_scale is not None:
            scale = np.maximum(shoulder_scale, self.cfg.epsilon)
            for i in range(0, min(seq.shape[1], 126), 3):
                seq[:, i:i+3] /= scale[:, np.newaxis]
            pose_start = 126
            for i in range(pose_start, min(seq.shape[1], pose_start + 100), 4):
                seq[:, i:i+3] /= scale[:, np.newaxis]

        parts = [seq]
        if self.cfg.add_velocity:
            vel = np.zeros_like(seq)
            vel[1:] = seq[1:] - seq[:-1]
            parts.append(vel)
        if self.cfg.add_acceleration:
            acc = np.zeros_like(seq)
            if seq.shape[0] > 2:
                acc[2:] = seq[2:] - 2*seq[1:-1] + seq[:-2]
            parts.append(acc)
        if self.cfg.add_distances:
            dists = self._compute_distances(seq)
            parts.append(dists)

        result = np.concatenate(parts, axis=-1)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def get_output_dim(self) -> int:
        dim = self.feature_dim
        if self.cfg.add_velocity:
            dim += self.feature_dim
        if self.cfg.add_acceleration:
            dim += self.feature_dim
        if self.cfg.add_distances:
            dim += 5  # hand-hand, hand-nose x2, hand-trunk x2
        return dim

    def _get_shoulder_ref(self, seq):
        """Extrai centro e distância entre ombros da pose."""
        T = seq.shape[0]
        # Pose starts after 2 hands: 2 * 21 * 3 = 126
        pose_start = 126
        ls_offset = pose_start + self.cfg.left_shoulder_pose_idx * 4
        rs_offset = pose_start + self.cfg.right_shoulder_pose_idx * 4

        if rs_offset + 3 > seq.shape[1]:
            return None, None

        ls = seq[:, ls_offset:ls_offset+3]
        rs = seq[:, rs_offset:rs_offset+3]
        center = (ls + rs) / 2.0
        dist = np.linalg.norm(ls - rs, axis=-1)
        median_dist = np.median(dist[dist > self.cfg.epsilon])
        if median_dist < self.cfg.epsilon:
            median_dist = 1.0
        dist = np.where(dist < self.cfg.epsilon, median_dist, dist)
        return center, dist

    def _compute_distances(self, seq):
        T = seq.shape[0]
        dists = np.zeros((T, 5), dtype=np.float32)
        # Left hand center (0:63) vs right hand center (63:126)
        if seq.shape[1] >= 126:
            lh = seq[:, :63].reshape(T, 21, 3).mean(axis=1)
            rh = seq[:, 63:126].reshape(T, 21, 3).mean(axis=1)
            dists[:, 0] = np.linalg.norm(lh - rh, axis=-1)
        return dists

    @staticmethod
    def _forward_fill(seq):
        for t in range(1, seq.shape[0]):
            zero_mask = np.all(np.abs(seq[t]) < 1e-10)
            if zero_mask:
                seq[t] = seq[t-1]
        return seq
