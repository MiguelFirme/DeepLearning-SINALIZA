#!/usr/bin/env python3
"""Mede amplitude de movimento por classe no split escolhido."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.inference.activity import pose_arm_motion
from ml.features.landmarks import LandmarkExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmarks", type=Path, default=Path("data/landmarks/minds_libras_school8"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/minds_libras_school8"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--still-video", type=Path,
                        help="Repete o primeiro quadro deste MP4 48 vezes para medir jitter")
    args = parser.parse_args()

    if args.still_video:
        cap = cv2.VideoCapture(str(args.still_video))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise FileNotFoundError(args.still_video)
        landmarks, masks = [], []
        with LandmarkExtractor() as extractor:
            for _ in range(48):
                x, mask = extractor.extract_frame(frame)
                landmarks.append(x)
                masks.append(mask)
        score = pose_arm_motion(np.stack(landmarks), np.stack(masks))
        print(f"QUADRO PARADO {args.still_video}: score={score:.3f}")
        return

    manifest = json.loads((args.landmarks / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((args.processed / "splits.json").read_text(encoding="utf-8"))
    by_label = defaultdict(list)
    for index in splits[args.split]:
        item = manifest[index]
        with np.load(args.landmarks / item["path"]) as data:
            landmarks = data["landmarks"]
            mask = data["mask"]
        selected = np.linspace(0, len(landmarks) - 1, 48, dtype=int)
        by_label[item["label"]].append(pose_arm_motion(landmarks[selected], mask[selected]))

    for label, values in sorted(by_label.items()):
        print(f"{label:12s} n={len(values):2d} min={min(values):.3f} "
              f"p50={np.median(values):.3f} max={max(values):.3f}")
    all_values = np.array([score for values in by_label.values() for score in values])
    print(f"TOTAL n={len(all_values)} quantis 0/5/10/25/50/90: "
          f"{np.percentile(all_values, [0, 5, 10, 25, 50, 90]).round(3).tolist()}")
    for threshold in (0.10, 0.15, 0.20, 0.25, 0.30):
        print(f"limiar {threshold:.2f}: {np.mean(all_values >= threshold):.1%} "
              "dos sinais com atividade suficiente")


if __name__ == "__main__":
    main()
