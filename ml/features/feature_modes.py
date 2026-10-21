"""Seleção das partes corporais disponíveis ao classificador."""
from __future__ import annotations

import numpy as np


FEATURE_MODES = {"full", "pose_face", "hands_only"}


def select_feature_mode(
    landmarks: np.ndarray,
    part_mask: np.ndarray | None,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Aplica o mesmo corte no treino e na inferência, antes da normalização."""
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode desconhecido: {feature_mode}")
    if feature_mode == "full":
        return landmarks, part_mask
    if landmarks.ndim != 2 or landmarks.shape[1] != 346 or part_mask is None or part_mask.shape != (landmarks.shape[0], 4):
        raise ValueError(f"{feature_mode} requer landmarks (T, 346) e máscara (T, 4)")

    selected = landmarks.copy()
    selected_mask = part_mask.copy()
    if feature_mode == "pose_face":
        selected[:, :126] = 0.0
        selected_mask[:, :2] = False
    else:
        selected[:, 126:] = 0.0
        selected_mask[:, 2:] = False
    return selected, selected_mask
