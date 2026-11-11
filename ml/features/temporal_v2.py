"""Reamostragem de landmarks brutos por tempo físico, sem comprimir o vídeo."""
from __future__ import annotations

import numpy as np

PART_SLICES = ((0, 63), (63, 126), (126, 226), (226, 346))


def resample_by_time(raw, mask, timestamps, end_time, target_fps, n_samples):
    """Retorna landmarks, máscara e grade uniforme terminando em end_time.

    Nas bordas repete a amostra extrema. Entre amostras, interpola somente
    partes válidas dos dois lados; caso contrário usa a detecção mais próxima
    sem inventar presença. Empates escolhem o vizinho anterior.
    """
    raw = np.asarray(raw, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool).copy()
    timestamps = np.asarray(timestamps, dtype=np.float64)
    n = len(raw)
    if n == 0 or raw.shape != (n, 346) or mask.shape != (n, 4) or timestamps.shape != (n,):
        raise ValueError("Landmarks/máscaras/tempos incompatíveis ou vazios")
    if (not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0)
            or not np.isfinite(end_time) or target_fps <= 0 or n_samples < 2):
        raise ValueError("Tempos devem ser crescentes; FPS e número de amostras positivos")
    grid = float(end_time) - np.arange(n_samples - 1, -1, -1) / target_fps
    right = np.clip(np.searchsorted(timestamps, grid, side="left"), 0, n - 1)
    left = np.maximum(right - 1, 0)
    left[grid >= timestamps[-1]] = n - 1
    right[grid <= timestamps[0]] = 0
    span = timestamps[right] - timestamps[left]
    alpha = np.divide(grid - timestamps[left], span, out=np.zeros(n_samples), where=span > 0)
    alpha = np.clip(alpha, 0, 1)
    nearest = np.where(alpha <= .5, left, right)
    out = np.zeros((n_samples, 346), dtype=np.float32)
    for side, (start, end) in enumerate(PART_SLICES):
        mask[:, side] &= np.isfinite(raw[:, start:end]).all(axis=1)
        valid = mask[left, side] & mask[right, side]
        values = raw[nearest, start:end].copy()
        values[valid] = ((1 - alpha[valid, None]) * raw[left[valid], start:end]
                         + alpha[valid, None] * raw[right[valid], start:end])
        values[~mask[nearest, side]] = 0
        out[:, start:end] = values
    return out, mask[nearest].copy(), grid
