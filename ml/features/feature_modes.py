"""Seleção das partes corporais disponíveis ao classificador."""
from __future__ import annotations

import numpy as np
from ml.features.landmarks import FACE_SELECTED_INDICES


FACE16_MEDIAPIPE_INDICES = (
    46, 65, 276, 295,                 # sobrancelhas
    33, 133, 160, 144, 362, 263, 387, 373,  # olhos
    13, 14, 78, 308,                  # boca
)
FACE16_KEEP_SLOTS = tuple(FACE_SELECTED_INDICES.index(index) for index in FACE16_MEDIAPIPE_INDICES)
FACE16_DROP_SLOTS = tuple(index for index in range(40) if index not in FACE16_KEEP_SLOTS)
FEATURE_MODES = {"full", "pose_face", "hands_only", "full_face16", "pose_face16"}


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
    if feature_mode in {"pose_face", "pose_face16"}:
        selected[:, :126] = 0.0
        selected_mask[:, :2] = False
    elif feature_mode == "hands_only":
        selected[:, 126:] = 0.0
        selected_mask[:, 2:] = False
    if feature_mode in {"pose_face16", "full_face16"}:
        selected[:, 226:346].reshape(-1, 40, 3)[:, FACE16_DROP_SLOTS, :] = 0.0
    return selected, selected_mask


def mask_normalized_face16(features: np.ndarray, feature_mode: str) -> np.ndarray:
    """Zera os pontos excluídos após centralizar pelos ombros e calcular velocidade."""
    if feature_mode not in {"pose_face16", "full_face16"}:
        return features
    if features.ndim != 2 or features.shape[1] != 697:
        raise ValueError(f"{feature_mode} requer features normalizadas (T, 697)")
    selected = features.copy()
    for start in (226, 346 + 226):
        selected[:, start:start + 120].reshape(-1, 40, 3)[:, FACE16_DROP_SLOTS, :] = 0.0
    return selected
