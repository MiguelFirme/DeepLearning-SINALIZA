"""Métricas conservadoras de qualidade para landmarks Holistic de 346 valores."""
from __future__ import annotations

import numpy as np


def active_interval(landmarks: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """Localiza janela de maior movimento dos punhos; fim exclusivo.

    Reutiliza o critério da auditoria: janela contígua de 40% dos quadros, com
    mínimo de oito. Não equivale a uma anotação linguística do início/fim.
    """
    n = len(landmarks)
    if n < 2:
        return 0, n
    pose = landmarks[:, 126:226].reshape(n, 25, 4)
    widths = np.linalg.norm(pose[:, 5, :2] - pose[:, 6, :2], axis=1)
    valid = mask[:, 2].astype(bool) & np.isfinite(widths) & (widths > .025)
    motion = np.zeros(n)
    if valid.any():
        width = np.median(widths[valid])
        for wrist, shoulder in ((9, 5), (10, 6)):
            rel = (pose[:, wrist, :2] - pose[:, shoulder, :2]) / width
            step = np.linalg.norm(np.diff(rel, axis=0), axis=1)
            step = np.where(valid[1:] & valid[:-1] & np.isfinite(step), step, 0)
            motion[1:] = np.maximum(motion[1:], step)
    window = min(n, max(8, int(round(n * .4))))
    start = int(np.argmax(np.convolve(motion, np.ones(window), mode="valid")))
    return start, start + window


def hand_quality(landmarks: np.ndarray, mask: np.ndarray) -> dict:
    """Estima a mão mais ativa pelo punho relativo ao ombro e mede cobertura.

    A janela de maior movimento é uma aproximação auditável do gesto, não uma
    anotação temporal manual. A função não altera nem reprova amostras.
    """
    if landmarks.ndim != 2 or landmarks.shape[1] != 346 or mask.shape != (len(landmarks), 4):
        raise ValueError("Esperado landmarks (T,346) e mask (T,4)")
    n = len(landmarks)
    if n < 2:
        raise ValueError("Vídeo com menos de dois quadros de landmarks")
    pose = landmarks[:, 126:226].reshape(n, 25, 4)
    shoulders = pose[:, [5, 6], :2]
    widths = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    valid_pose = mask[:, 2].astype(bool) & np.isfinite(widths) & (widths > 0.025)
    width = float(np.median(widths[valid_pose])) if valid_pose.any() else 0.0
    motion = np.zeros((n, 2), dtype=np.float32)
    if width > 0:
        for side, (wrist, shoulder) in enumerate(((9, 5), (10, 6))):
            rel = (pose[:, wrist, :2] - pose[:, shoulder, :2]) / width
            steps = np.linalg.norm(np.diff(rel, axis=0), axis=1)
            valid_pairs = valid_pose[1:] & valid_pose[:-1] & np.isfinite(steps)
            motion[1:, side] = np.where(valid_pairs, steps, 0.0)
    scores = np.percentile(motion, 90, axis=0)
    active_side = int(np.argmax(scores))
    active_name = ("left", "right")[active_side]
    # Uma janela contígua evita chamar ruído isolado de gesto inteiro.
    total_motion = motion.max(axis=1)
    start, end = active_interval(landmarks, mask)
    active_detected = mask[start:end, active_side].astype(bool)
    active_coverage = float(active_detected.mean())
    full_coverage = float(mask[:, active_side].mean())
    both_coverage = float(np.mean(mask[start:end, 0] & mask[start:end, 1]))

    hand = landmarks[:, active_side * 63:(active_side + 1) * 63].reshape(n, 21, 3)
    hand_xy = hand[:, :, :2]
    finite = np.isfinite(hand_xy).all(axis=(1, 2))
    in_frame = ((hand_xy >= 0.0) & (hand_xy <= 1.0)).all(axis=(1, 2))
    active_valid = mask[start:end, active_side].astype(bool)
    xy_ok = float(np.mean((finite & in_frame)[start:end][active_valid])) if active_valid.any() else 0.0
    visible = mask[:, active_side].astype(bool)
    jumps = np.linalg.norm(np.diff(hand_xy[:, 0], axis=0), axis=1)
    pairs = visible[1:] & visible[:-1] & np.isfinite(jumps)
    jump_fraction = float(np.mean(jumps[pairs] > max(0.15, width))) if pairs.any() else 0.0
    near_edge = (((hand_xy[:, :, 0] < 0.03) | (hand_xy[:, :, 0] > 0.97) |
                  (hand_xy[:, :, 1] < 0.03) | (hand_xy[:, :, 1] > 0.97)).any(axis=1))
    edge_fraction = float(np.mean(near_edge[start:end][active_valid])) if active_valid.any() else 0.0
    tail = max(2, n // 10)
    boundary_motion = float(max(np.mean(total_motion[:tail]), np.mean(total_motion[-tail:])))
    central_motion = float(np.mean(total_motion[start:end]))
    return {
        "frames": n, "active_hand": active_name, "active_start": start,
        "active_end": end, "active_hand_coverage": active_coverage,
        "full_hand_coverage": full_coverage, "both_hands_active_coverage": both_coverage,
        "active_hand_in_frame": xy_ok, "hand_wrist_jump_fraction": jump_fraction,
        "active_hand_edge_fraction": edge_fraction,
        "pose_coverage": float(mask[:, 2].mean()),
        "face_coverage": float(mask[:, 3].mean()),
        "active_motion_score": float(scores[active_side]),
        "central_motion": central_motion,
        "boundary_motion": boundary_motion,
        "shoulder_width_px_fraction": width,
    }
