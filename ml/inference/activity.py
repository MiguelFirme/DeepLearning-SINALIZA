"""Mede atividade de braços em landmarks brutos, antes da classificação."""
from __future__ import annotations

import numpy as np


def pose_arm_motion(landmarks: np.ndarray, part_mask: np.ndarray) -> float:
    """Amplitude robusta dos cotovelos/pulsos em unidades de largura dos ombros.

    Remove translação do corpo usando o ombro do mesmo lado. Retorna zero
    quando a pose válida é insuficiente para medir atividade.
    """
    if landmarks.ndim != 2 or landmarks.shape[1] != 346:
        raise ValueError(f"Esperado landmarks (T, 346), recebido {landmarks.shape}")
    if part_mask.shape != (landmarks.shape[0], 4):
        raise ValueError(f"Esperado mask (T, 4), recebido {part_mask.shape}")
    valid = part_mask[:, 2].astype(bool)
    if valid.sum() < max(8, landmarks.shape[0] // 2):
        return 0.0

    pose = landmarks[:, 126:226].reshape(-1, 25, 4)
    shoulders = pose[:, [5, 6], :2]
    width = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    valid &= np.isfinite(width) & (width > 0.025)
    if valid.sum() < max(8, landmarks.shape[0] // 2):
        return 0.0
    scale = float(np.median(width[valid]))

    scores = []
    for joint, shoulder in ((7, 5), (8, 6), (9, 5), (10, 6)):
        relative = (pose[valid, joint, :2] - pose[valid, shoulder, :2]) / scale
        if not np.isfinite(relative).all():
            continue
        span = np.percentile(relative, 90, axis=0) - np.percentile(relative, 10, axis=0)
        scores.append(float(np.linalg.norm(span)))
    return max(scores, default=0.0)
