#!/usr/bin/env python3
"""Audita cobertura/continuidade das mãos e qualidade visual dos 25 sinais."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data.quality import hand_quality


def raw_video(sample: dict, raw_dir: Path) -> Path:
    source, video_id = sample["video_id"].split(":", 1)
    if source == "minds":
        return raw_dir / "minds_libras" / video_id
    if source == "novas":
        return raw_dir / video_id
    raise ValueError(f"Origem desconhecida: {source}")


def video_quality(path: Path, samples: int) -> dict:
    if not path.is_file():
        return {"video_exists": False, "decoded_samples": 0, "blur_score": None,
                "dark_fraction": None, "reported_frames": 0}
    if samples == 0:
        return {"video_exists": True, "decoded_samples": 0, "blur_score": None,
                "dark_fraction": None, "reported_frames": None}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"video_exists": True, "decoded_samples": 0, "blur_score": None,
                "dark_fraction": None, "reported_frames": 0}
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    blur, dark = [], []
    try:
        for index in np.linspace(0, max(0, count - 1), samples, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (320, max(1, round(h * 320 / w))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            dark.append(float(gray.mean() < 35))
    finally:
        cap.release()
    return {"video_exists": True, "decoded_samples": len(blur),
            "blur_score": float(np.median(blur)) if blur else None,
            "dark_fraction": float(np.mean(dark)) if dark else None,
            "reported_frames": count}


def audit(data_dir: Path, processed_dir: Path, raw_dir: Path,
          output_dir: Path, video_samples: int) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((processed_dir / "splits.json").read_text(encoding="utf-8"))
    owner = {index: split for split, indices in splits.items() for index in indices}
    if len(owner) != len(manifest) or len(manifest) != 670:
        raise ValueError("Este relatório requer o manifesto misto completo de 670 vídeos")
    rows = []
    for index, sample in enumerate(manifest):
        npz = data_dir / sample["path"]
        with np.load(npz) as data:
            features = hand_quality(data["landmarks"], data["mask"])
        video = raw_video(sample, raw_dir)
        visual = video_quality(video, video_samples)
        flags = []
        if features["active_hand_coverage"] < 0.5:
            flags.append("active_hand_coverage_lt_50pct")
        if features["active_hand_in_frame"] < 0.8:
            flags.append("active_hand_out_of_frame")
        if features["hand_wrist_jump_fraction"] > 0.1:
            flags.append("hand_wrist_jumps")
        if features["active_hand_edge_fraction"] > 0.3:
            flags.append("hand_near_frame_edge")
        if features["pose_coverage"] < 0.8:
            flags.append("pose_missing")
        if not visual["video_exists"]:
            flags.append("video_missing")
        if video_samples and visual["decoded_samples"] < video_samples:
            flags.append("video_decode_incomplete")
        if visual["blur_score"] is not None and visual["blur_score"] < 20:
            flags.append("possibly_blurry")
        if visual["dark_fraction"] is not None and visual["dark_fraction"] > 0.5:
            flags.append("possibly_dark")
        # Movimento alto nas bordas é alerta de possível corte, não prova.
        if features["boundary_motion"] > max(0.15, features["central_motion"] * 1.5):
            flags.append("possible_temporal_cut")
        rows.append({"split": owner[index], "origin": sample["origin"],
                     "label": sample["label"], "signer": sample["signer"],
                     "path": sample["path"], "video": str(video),
                     **features, **visual, "flags": ";".join(flags),
                     "review_decision": "pending" if flags else "not_flagged",
                     "reviewer": "", "review_reason": ""})
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for name, selected in (("quality_all.csv", rows),
                           ("review_queue.csv", [row for row in rows if row["flags"]])):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    summary = {
        "scope": "15 selected MINDS classes + 10 new recorded classes",
        "total_videos": len(rows), "flagged_videos": sum(bool(row["flags"]) for row in rows),
        "flags": dict(Counter(flag for row in rows for flag in row["flags"].split(";") if flag)),
        "by_split": {
            name: {"videos": len(group),
                   "flagged": sum(bool(row["flags"]) for row in group),
                   "median_active_hand_coverage": float(np.median(
                       [row["active_hand_coverage"] for row in group]))}
            for name, group in by_split.items()
        },
        "thresholds": {"active_hand_coverage": 0.5, "active_hand_in_frame": 0.8,
                       "wrist_jump_fraction": 0.1, "edge_fraction": 0.3,
                       "pose_coverage": 0.8, "blur_score": 20},
        "interpretation": "Flags require human video review; no sample was changed or excluded.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/landmarks/mixed_25"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/mixed_25"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/quality_mixed25"))
    parser.add_argument("--video-samples", type=int, default=0,
                        help="Quadros a decodificar por vídeo; 0 audita apenas landmarks e existência do MP4")
    args = parser.parse_args()
    if args.video_samples < 0:
        parser.error("--video-samples precisa ser zero ou positivo")
    print(json.dumps(audit(args.data_dir, args.processed_dir, args.raw_dir,
                           args.output_dir, args.video_samples), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
