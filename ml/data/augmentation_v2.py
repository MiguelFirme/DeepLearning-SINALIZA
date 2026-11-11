"""Aumentos v2 em landmarks brutos com máscaras e relógio sincronizados."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.features.landmarks import FACE_SELECTED_INDICES, POSE_SELECTED_INDICES

FACE_PAIRS = ((46, 276), (52, 282), (53, 283), (55, 285), (65, 295),
              (33, 263), (133, 362), (160, 387), (144, 373), (98, 327),
              (78, 308), (82, 312), (87, 317), (234, 454), (127, 356))
FACE_CENTRAL = {1, 2, 13, 14, 0, 10, 152, 168, 6}
POSE_PAIRS = ((2, 5), (7, 8), (11, 12), (13, 14), (15, 16), (17, 18),
              (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30))


def mirror_raw_v2(raw, mask):
    """Espelha x, troca lateralidade e zera ponto facial sem correspondente (267)."""
    result, mask = np.asarray(raw).copy(), np.asarray(mask, dtype=bool).copy()
    result[:, :63], result[:, 63:126] = raw[:, 63:126], raw[:, :63]
    mask[:, [0, 1]] = mask[:, [1, 0]]
    pose = result[:, 126:226].reshape(-1, 25, 4)
    for left, right in POSE_PAIRS:
        a, b = POSE_SELECTED_INDICES.index(left), POSE_SELECTED_INDICES.index(right)
        pose[:, [a, b]] = pose[:, [b, a]]
    face = result[:, 226:346].reshape(-1, 40, 3)
    mapping = {i: i for i in FACE_CENTRAL}
    for a, b in FACE_PAIRS:
        mapping[a], mapping[b] = b, a
    original = face.copy()
    for i, point in enumerate(FACE_SELECTED_INDICES):
        partner = mapping.get(point)
        face[:, i] = original[:, FACE_SELECTED_INDICES.index(partner)] if partner in FACE_SELECTED_INDICES else 0
    for start, end, step in ((0, 126, 3), (126, 226, 4), (226, 346, 3)):
        result[:, start:end:step] = 1 - result[:, start:end:step]
    # A inversão não pode transformar ausência em um ponto x=1.
    for side, (start, end) in enumerate(((0, 63), (63, 126), (126, 226), (226, 346))):
        result[~mask[:, side], start:end] = 0
    for i, point in enumerate(FACE_SELECTED_INDICES):
        if point not in mapping:
            face[:, i] = 0
    return result, mask


@dataclass
class AugmentationV2Config:
    mirror_prob: float = .5
    speed_range: tuple = (.7, 1.3)
    scale_range: tuple = (.85, 1.15)
    rotate_degrees: float = 10.
    translate: float = .05
    noise_std: float = .005
    max_frame_drop: float = .3
    hand_missing_prob: float = .2

    def __post_init__(self):
        for p in (self.mirror_prob, self.hand_missing_prob, self.max_frame_drop):
            if not 0 <= p <= 1:
                raise ValueError("Probabilidades devem estar entre zero e um")
        if self.max_frame_drop > .9 or min(self.speed_range) <= 0 or min(self.scale_range) <= 0:
            raise ValueError("Escala/velocidade positivas e descarte máximo de 90%")


class RawAugmentorV2:
    def __init__(self, cfg=None):
        self.cfg = cfg or AugmentationV2Config()

    def __call__(self, raw, mask, times, rng, width=1, height=1):
        raw, mask, times = raw.copy(), mask.copy(), times.copy()
        cfg = self.cfg
        if rng.random() < cfg.mirror_prob:
            raw, mask = mirror_raw_v2(raw, mask)
        scale, angle = rng.uniform(*cfg.scale_range), np.deg2rad(rng.uniform(-cfg.rotate_degrees, cfg.rotate_degrees))
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        shift = rng.uniform(-cfg.translate, cfg.translate, 2)
        aspect = width / height
        for start, end, step in ((0, 126, 3), (126, 226, 4), (226, 346, 3)):
            points = raw[:, start:end].reshape(len(raw), -1, step)
            present = np.any(points[:, :, :3] != 0, axis=-1)
            xy = (points[:, :, :2] - .5) * [aspect, 1]
            points[:, :, :2] = (xy @ rotation.T * scale + shift) / [aspect, 1] + .5
            points[:, :, 2] *= scale
            points[:, :, :3] += rng.normal(0, cfg.noise_std, points[:, :, :3].shape)
            points[:, :, :3][~present] = 0
        for side, (start, end) in enumerate(((0, 63), (63, 126), (126, 226), (226, 346))):
            raw[~mask[:, side], start:end] = 0
        if len(raw) > 2 and rng.random() < cfg.hand_missing_prob:
            side = int(rng.integers(2))
            span = int(rng.integers(1, max(2, len(raw) // 3 + 1)))
            start = int(rng.integers(0, len(raw) - span + 1))
            raw[start:start + span, side * 63:(side + 1) * 63] = 0
            mask[start:start + span, side] = False
        times = (times - times[0]) * rng.uniform(*cfg.speed_range) + times[0]
        count = int(rng.uniform(0, cfg.max_frame_drop) * len(raw))
        if count and len(raw) > 2:
            keep = np.ones(len(raw), bool)
            keep[rng.choice(np.arange(1, len(raw) - 1), min(count, len(raw) - 2), replace=False)] = False
            raw, mask, times = raw[keep], mask[keep], times[keep]
        return raw, mask, times
