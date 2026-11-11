"""Features v2 compartilhadas: coordenadas globais, mãos locais e velocidade/s."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_DIM_V2 = 725
GLOBAL_DIM_V2 = 297
PART_SLICES_V2 = ((0, 63), (63, 126), (126, 177), (177, 297))


def prepare_features_v2(raw, part_mask, timestamps, width, height, cfg=None, end_time=None):
    """Entrada única de janela para dataset e backend: tempo antes de velocidade."""
    from ml.features.temporal_v2 import resample_by_time
    cfg = cfg or PipelineV2Config()
    sampled, sampled_mask, grid = resample_by_time(
        raw, part_mask, timestamps, timestamps[-1] if end_time is None else end_time,
        cfg.target_fps, cfg.n_samples,
    )
    features = build_features_v2(sampled, sampled_mask, grid, width, height, cfg)
    # A validade temporal indica presença de ao menos uma parte. Lacunas curtas
    # das mãos já são preenchidas pelo construtor; pose/face preservam a presença.
    return features, sampled_mask.any(axis=1), sampled_mask


@dataclass
class PipelineV2Config:
    target_fps: float = 15.0
    window_seconds: float = 3.2
    max_gap: int = 5
    epsilon: float = 1e-6

    def __post_init__(self):
        if self.target_fps <= 0 or self.window_seconds <= 0 or self.max_gap < 0:
            raise ValueError("FPS, duração e tamanho de lacuna inválidos")
        if self.epsilon <= 0 or self.n_samples < 2:
            raise ValueError("Épsilon positivo e ao menos duas amostras são necessários")

    @property
    def n_samples(self) -> int:
        return int(round(self.target_fps * self.window_seconds))


def fill_hand_gaps(raw, mask, max_gap):
    """Preenche somente lacunas internas, limitadas por duas detecções reais."""
    raw, mask = raw.copy(), mask.copy()
    for side in (0, 1):
        valid = np.flatnonzero(mask[:, side])
        for left, right in zip(valid[:-1], valid[1:]):
            gap = right - left - 1
            if 0 < gap <= max_gap:
                alpha = np.arange(1, gap + 1, dtype=np.float32)[:, None] / (gap + 1)
                region = slice(side * 63, (side + 1) * 63)
                raw[left + 1:right, region] = (1 - alpha) * raw[left, region] + alpha * raw[right, region]
                mask[left + 1:right, side] = True
    return raw, mask


def build_features_v2(raw, part_mask, timestamps, width, height, cfg=None):
    """Constrói (T,725) em grade uniforme; reamostrar antes desta função.

    A ordem é: aspecto, remoção de pernas/visibilidade, lacunas, referência dos
    ombros, mãos locais, distâncias e velocidade global por segundo. Não altera
    entradas. As 126 coordenadas locais não recebem uma segunda velocidade.
    """
    cfg = cfg or PipelineV2Config()
    raw = np.asarray(raw, dtype=np.float32).copy()
    mask = np.asarray(part_mask, dtype=bool).copy()
    times = np.asarray(timestamps, dtype=np.float64)
    n = len(raw)
    if raw.shape != (n, 346) or mask.shape != (n, 4) or times.shape != (n,) or n == 0:
        raise ValueError("Esperado raw (T,346), máscara (T,4) e timestamps (T,), T > 0")
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Largura e altura devem ser positivas")
    if not np.isfinite(times).all() or (n > 1 and not np.allclose(np.diff(times), 1 / cfg.target_fps, atol=1e-6)):
        raise ValueError("Reamostre os timestamps na grade de FPS alvo antes das features")
    # Valores não finitos invalidam a parte inteira; nunca viram detecção falsa.
    for side, (start, end) in enumerate(((0, 63), (63, 126), (126, 226), (226, 346))):
        mask[:, side] &= np.isfinite(raw[:, start:end]).all(axis=1)
    raw = np.nan_to_num(raw, nan=0, posinf=0, neginf=0)
    raw[:, 0:126:3] *= width / height
    raw[:, 126:226:4] *= width / height
    raw[:, 226:346:3] *= width / height
    # As posições 0..16 da pose incluem nariz, ombros, braços e dedos.
    pose = raw[:, 126:226].reshape(n, 25, 4)[:, :17, :3].copy()
    compact = np.concatenate((raw[:, :126], pose.reshape(n, 51), raw[:, 226:]), axis=1)
    compact, mask = fill_hand_gaps(compact, mask, cfg.max_gap)
    hands = compact[:, :126].reshape(n, 2, 21, 3).copy()
    pose = compact[:, 126:177].reshape(n, 17, 3)
    center = (pose[:, 5] + pose[:, 6]) / 2
    scale = np.linalg.norm(pose[:, 5] - pose[:, 6], axis=1)
    valid = mask[:, 2] & (scale > cfg.epsilon)
    if valid.any():
        center[~valid] = np.median(center[valid], axis=0)
        scale[~valid] = np.median(scale[valid])
    else:
        center[:] = 0
        scale[:] = 1
    global_xyz = compact.reshape(n, -1, 3)
    global_xyz = (global_xyz - center[:, None]) / np.maximum(scale[:, None, None], cfg.epsilon)
    global_xyz = global_xyz.reshape(n, GLOBAL_DIM_V2)
    for side, (start, end) in enumerate(PART_SLICES_V2):
        global_xyz[~mask[:, side], start:end] = 0
    local = hands - hands[:, :, :1]
    hand_scale = np.linalg.norm(local[:, :, 9], axis=-1)
    local /= np.maximum(hand_scale[:, :, None, None], cfg.epsilon)
    for side in (0, 1):
        local[~mask[:, side], side] = 0
        local[hand_scale[:, side] <= cfg.epsilon, side] = 0
    distances = np.zeros((n, 5), dtype=np.float32)
    hand_centers = global_xyz[:, :126].reshape(n, 2, 21, 3).mean(axis=2)
    nose = global_xyz[:, 126:129]
    pairs = ((0, 1), (0, 2), (1, 2), (0, 2), (1, 2))
    vectors = (hand_centers[:, 0] - hand_centers[:, 1], hand_centers[:, 0] - nose,
               hand_centers[:, 1] - nose, hand_centers[:, 0], hand_centers[:, 1])
    for i, ((a, b), delta) in enumerate(zip(pairs, vectors)):
        ok = mask[:, a] & mask[:, b]
        distances[ok, i] = np.linalg.norm(delta[ok], axis=1)
    velocity = np.zeros_like(global_xyz)
    velocity[1:] = np.diff(global_xyz, axis=0) * cfg.target_fps
    for side, (start, end) in enumerate(PART_SLICES_V2):
        velocity[1:, start:end][~(mask[1:, side] & mask[:-1, side])] = 0
    return np.concatenate((global_xyz, local.reshape(n, 126), distances, velocity), axis=1).astype(np.float32)
