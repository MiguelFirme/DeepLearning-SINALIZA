#!/usr/bin/env python3
"""Compara o Holistic em cache com MediaPipe Hands em quadros dos mesmos videos.

Uso (Git Bash):
  ./.venv/Scripts/python.exe scripts/audit_hand_detector.py \
    --videos data/raw/minds_libras/16MedoSinalizador01-1.mp4 \
    --landmarks-dir data/landmarks/minds_libras_school5 \
    --output logs/minds_hand_detector_audit.json

Nao altera os landmarks nem utiliza rotulos para selecionar o detector.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data.minds_libras import parse_video_name


def sample_indices(frame_count: int, samples: int) -> list[int]:
    if frame_count <= 0 or samples <= 0:
        return []
    return sorted(set(np.linspace(0, frame_count - 1, min(frame_count, samples), dtype=int).tolist()))


def audit_video(video: Path, landmarks_dir: Path, hands, samples: int,
                preview_dir: Path | None = None) -> dict:
    cache = landmarks_dir / f"{video.stem}.npz"
    if not cache.is_file():
        raise FileNotFoundError(cache)
    with np.load(cache, allow_pickle=False) as data:
        old_mask = data["mask"][:, :2].astype(bool)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir {video}")
    rows = []
    previews_saved = 0
    try:
        for index in sample_indices(len(old_mask), samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            detected = result.multi_hand_landmarks or []
            # Coordenadas fora da imagem podem indicar deteccao instavel.
            in_frame = [
                all(0 <= lm.x <= 1 and 0 <= lm.y <= 1 for lm in hand.landmark)
                for hand in detected
            ]
            rows.append({
                "frame": index,
                "holistic_hands": int(old_mask[index].sum()),
                "hands_detector_hands": len(detected),
                "hands_all_points_in_frame": in_frame,
            })
            if (preview_dir is not None and len(detected) > int(old_mask[index].sum())
                    and previews_saved < 4):
                preview_dir.mkdir(parents=True, exist_ok=True)
                overlay = frame.copy()
                for hand in detected:
                    mp.solutions.drawing_utils.draw_landmarks(
                        overlay, hand, mp.solutions.hands.HAND_CONNECTIONS)
                cv2.imwrite(str(preview_dir / f"{video.stem}_frame{index:04d}.jpg"), overlay)
                previews_saved += 1
    finally:
        cap.release()
    if not rows:
        raise RuntimeError(f"Nenhum quadro lido: {video}")
    old = np.array([r["holistic_hands"] for r in rows])
    new = np.array([r["hands_detector_hands"] for r in rows])
    return {
        "video": video.name,
        **parse_video_name(video.name),
        "sampled_frames": len(rows),
        "holistic_any": int((old > 0).sum()),
        "holistic_both": int((old == 2).sum()),
        "hands_any": int((new > 0).sum()),
        "hands_both": int((new == 2).sum()),
        "hands_gain_frames": int((new > old).sum()),
        "hands_loss_frames": int((new < old).sum()),
        "hands_out_of_frame_frames": sum(not all(r["hands_all_points_in_frame"]) for r in rows),
        "previews_saved": previews_saved,
        "frames": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", type=Path, required=True)
    parser.add_argument("--landmarks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--samples-per-video", type=int, default=40)
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.3)
    args = parser.parse_args()
    if not 0 < args.min_hand_detection_confidence <= 1:
        parser.error("O limiar deve estar entre 0 (exclusivo) e 1")
    if args.samples_per_video < 1:
        parser.error("--samples-per-video deve ser positivo")

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=args.min_hand_detection_confidence,
    ) as hands:
        videos = [audit_video(path, args.landmarks_dir, hands, args.samples_per_video,
                              args.preview_dir)
                  for path in args.videos]
    totals = {key: sum(v[key] for v in videos) for key in (
        "sampled_frames", "holistic_any", "holistic_both", "hands_any", "hands_both",
        "hands_gain_frames", "hands_loss_frames", "hands_out_of_frame_frames")}
    report = {
        "detector": "mediapipe.solutions.hands.Hands",
        "static_image_mode": True,
        "min_hand_detection_confidence": args.min_hand_detection_confidence,
        "comparison_note": "Taxa de deteccao nao demonstra que os pontos estao corretos; inspecao visual e validacao sao necessarias antes de reextrair.",
        "totals": totals,
        "videos": videos,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"totals": totals, "videos": [
        {key: v[key] for key in ("video", "sampled_frames", "holistic_any", "holistic_both",
                              "hands_any", "hands_both", "hands_gain_frames", "hands_loss_frames")}
        for v in videos]}, ensure_ascii=False, indent=2))
    print(f"Relatorio: {args.output}")


if __name__ == "__main__":
    main()
