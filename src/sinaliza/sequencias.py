"""Operações com sequências temporais de landmarks."""

from __future__ import annotations

import numpy as np


def redimensionar_sequencia(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Interpola uma sequência para que tenha exatamente ``target_length`` frames."""
    if target_length <= 0:
        raise ValueError("target_length deve ser maior que zero")

    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("sequence deve possuir o formato (frames, features)")
    if values.shape[0] == target_length:
        return values.copy()
    if values.shape[0] == 1:
        return np.repeat(values, target_length, axis=0)

    old_positions = np.linspace(0.0, 1.0, values.shape[0])
    new_positions = np.linspace(0.0, 1.0, target_length)
    resized = np.empty((target_length, values.shape[1]), dtype=np.float32)
    for feature_index in range(values.shape[1]):
        resized[:, feature_index] = np.interp(
            new_positions, old_positions, values[:, feature_index]
        )
    return resized
