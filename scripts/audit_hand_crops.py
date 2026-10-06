#!/usr/bin/env python3
"""Audita deteccao de maos em recortes de pulso guiados por pose.

Compara nos mesmos frames do relatorio de `audit_hand_detector.py`.
Nao modifica landmarks, manifests nem checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.features.landmarks import POSE_SELECTED_INDICES

POSE_OFFSET = 2 * 21 * 3
POSE_LENGTH = len(POSE_SELECTED_INDICES) * 4
JOINTS = {
    "left": (POSE_SELECTED_INDICES.index(13), POSE_SELECTED_INDICES.index(15)),
    "right": (POSE_SELECTED_INDICES.index(14), POSE_SELECTED_INDICES.index(16)),
}


def wrist_crop(pose: np.ndarray, side: str, width: int, height: int,
               min_visibility: float = 0.5) -> tuple[int, int, int, int] | None:
    """Retorna (x0,y0,x1,y1), ou None se pulso/cotovelo nao confiaveis."""
    elbow_i, wrist_i = JOINTS[side]
    elbow, wrist = pose[elbow_i], pose[wrist_i]
    if not (np.isfinite(elbow).all() and np.isfinite(wrist).all()):
        return None
    if min(elbow[3], wrist[3]) < min_visibility:
        return None
    if not (0 <= wrist[0] <= 1 and 0 <= wrist[1] <= 1):
        return None
    ex, ey = float(elbow[0] * width), float(elbow[1] * height)
    wx, wy = float(wrist[0] * width), float(wrist[1] * height)
    forearm = math.hypot(wx - ex, wy - ey)
    if forearm < 12:
        return None
    size = int(np.clip(3.2 * forearm, 160, 0.65 * min(width, height)))
    cx, cy = wx + 0.35 * (wx - ex), wy + 0.35 * (wy - ey)
    x0, y0 = max(0, round(cx - size / 2)), max(0, round(cy - size / 2))
    x1, y1 = min(width, round(cx + size / 2)), min(height, round(cy + size / 2))
    if x1 - x0 < 80 or y1 - y0 < 80:
        return None
    return x0, y0, x1, y1


def detect_for_wrist(frame: np.ndarray, pose: np.ndarray, side: str,
                     detector) -> tuple[dict | None, str]:
    height, width = frame.shape[:2]
    box = wrist_crop(pose, side, width, height)
    if box is None:
        return None, "pose_unavailable"
    x0, y0, x1, y1 = box
    rgb = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = detector.process(rgb)
    candidates = []
    for hand in result.multi_hand_landmarks or []:
        points = np.array([
            (x0 + lm.x * (x1 - x0), y0 + lm.y * (y1 - y0))
            for lm in hand.landmark
        ], dtype=float)
        wrist_xy = pose[JOINTS[side][1], :2] * (width, height)
        distance = float(np.linalg.norm(points[0] - wrist_xy))
        candidates.append((distance, points))
    if not candidates:
        return None, "not_detected"
    distance, points = min(candidates, key=lambda item: item[0])
    if distance > 0.45 * max(x1 - x0, y1 - y0):
        return None, "too_far_from_pose_wrist"
    return {"side": side, "box": list(box), "points": points,
            "distance_to_pose_wrist_px": round(distance, 1)}, "detected"


def distinct_candidates(candidates: list[dict], width: int, height: int) -> list[dict]:
    """Remove a mesma mao capturada por ambos os recortes."""
    if len(candidates) == 2:
        wrist_distance = np.linalg.norm(candidates[0]["points"][0] -
                                        candidates[1]["points"][0])
        if wrist_distance < 0.06 * min(width, height):
            return [min(candidates, key=lambda c: c["distance_to_pose_wrist_px"])]
    return candidates


def save_preview(path: Path, frame: np.ndarray, pose: np.ndarray,
                 candidates: list[dict], statuses: dict, old_count: int,
                 full_count: int) -> None:
    image = frame.copy()
    colors = {"left": (0, 255, 255), "right": (255, 180, 0)}
    for side in JOINTS:
        box = wrist_crop(pose, side, frame.shape[1], frame.shape[0])
        if box is not None:
            cv2.rectangle(image, box[:2], box[2:], colors[side], 2)
    for item in candidates:
        points = item["points"].round().astype(int)
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 4, colors[item["side"]], -1)
    cv2.putText(image, f"Holistic {old_count} | Full Hands {full_count} | Crop {len(candidates)}",
                (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(image, f"L: {statuses['left']}  R: {statuses['right']}",
                (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Falha ao salvar {path}")


def audit_video(reference: dict, video_dir: Path, landmarks_dir: Path,
                detector, preview_dir: Path) -> dict:
    name = reference["video"]
    video = video_dir / name
    cache = landmarks_dir / f"{video.stem}.npz"
    with np.load(cache, allow_pickle=False) as data:
        landmarks = data["landmarks"]
        mask = data["mask"]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir {video}")
    rows = []
    previews = {"gain": 0, "loss": 0, "miss": 0}
    try:
        for ref in reference["frames"]:
            index = ref["frame"]
            if index >= len(landmarks):
                raise ValueError(f"Frame {index} ausente no cache {cache}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Nao foi possivel ler frame {index} de {video}")
            pose = landmarks[index, POSE_OFFSET:POSE_OFFSET + POSE_LENGTH].reshape(-1, 4)
            candidates, statuses = [], {}
            for side in JOINTS:
                hand, status = detect_for_wrist(frame, pose, side, detector)
                statuses[side] = status
                if hand is not None:
                    candidates.append(hand)
            candidates = distinct_candidates(candidates, frame.shape[1], frame.shape[0])
            old_count = int(mask[index, :2].sum())
            if old_count != ref["holistic_hands"]:
                raise ValueError(f"Mascara e relatorio divergem: {name}, frame {index}")
            crop_count = len(candidates)
            full_count = ref["hands_detector_hands"]
            if crop_count > old_count:
                kind = "gain"
            elif crop_count < old_count:
                kind = "loss"
            elif old_count < 2:
                kind = "miss"
            else:
                kind = None
            if kind and previews[kind] < 4:
                save_preview(preview_dir / kind / f"{video.stem}_frame{index:04d}.jpg",
                             frame, pose, candidates, statuses, old_count, full_count)
                previews[kind] += 1
            rows.append({"frame": index, "holistic_hands": old_count,
                         "full_hands": full_count, "crop_hands": crop_count,
                         "crop_status": statuses,
                         "crop_wrist_distances_px": {
                             c["side"]: c["distance_to_pose_wrist_px"] for c in candidates}})
    finally:
        cap.release()
    old = np.array([r["holistic_hands"] for r in rows])
    full = np.array([r["full_hands"] for r in rows])
    crop = np.array([r["crop_hands"] for r in rows])
    return {"video": name, "label": reference["label"], "signer": reference["signer"],
            "sampled_frames": len(rows), "holistic_any": int((old > 0).sum()),
            "holistic_both": int((old == 2).sum()), "full_any": int((full > 0).sum()),
            "full_both": int((full == 2).sum()), "crop_any": int((crop > 0).sum()),
            "crop_both": int((crop == 2).sum()), "crop_gain_frames": int((crop > old).sum()),
            "crop_loss_frames": int((crop < old).sum()),
            "pose_unavailable_sides": sum(s == "pose_unavailable" for r in rows
                                          for s in r["crop_status"].values()),
            "previews_saved": previews, "frames": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--landmarks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.3)
    args = parser.parse_args()
    if not 0 < args.min_hand_detection_confidence <= 1:
        parser.error("O limiar deve estar entre 0 (exclusivo) e 1")
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                  min_detection_confidence=args.min_hand_detection_confidence) as detector:
        videos = [audit_video(v, args.video_dir, args.landmarks_dir, detector,
                              args.preview_dir) for v in reference["videos"]]
    keys = ("sampled_frames", "holistic_any", "holistic_both", "full_any", "full_both",
            "crop_any", "crop_both", "crop_gain_frames", "crop_loss_frames",
            "pose_unavailable_sides")
    totals = {key: sum(v[key] for v in videos) for key in keys}
    result = {"method": "MediaPipe Hands on pose wrist crops",
              "min_hand_detection_confidence": args.min_hand_detection_confidence,
              "reference_report": str(args.reference_report), "totals": totals,
              "caution": "Ganho de contagem nao prova mao correta; inspecione as imagens antes de qualquer reextracao.",
              "videos": videos}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"totals": totals, "videos": [
        {k: v[k] for k in ("video", "sampled_frames", "holistic_both", "full_both",
                          "crop_both", "crop_gain_frames", "crop_loss_frames",
                          "pose_unavailable_sides")}
        for v in videos]}, ensure_ascii=False, indent=2))
    print(f"Relatorio: {args.output}\nPrevias: {args.preview_dir}")


if __name__ == "__main__":
    main()
