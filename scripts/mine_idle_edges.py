#!/usr/bin/env python3
"""Find review candidates for no-sign segments at video boundaries.

The CSV is an audit queue, not negative labels. No NPZ is exported here.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_mixed25_quality import raw_video


def motion_curve(landmarks: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(landmarks)
    pose = landmarks[:, 126:226].reshape(n, 25, 4)
    shoulders = pose[:, [5, 6], :2]
    widths = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    valid = mask[:, 2].astype(bool) & np.isfinite(widths) & (widths > 0.025)
    if not valid.any():
        return np.zeros(n, dtype=np.float32), valid
    width = np.median(widths[valid])
    relative = np.stack(((pose[:, 9, :2] - pose[:, 5, :2]) / width,
                         (pose[:, 10, :2] - pose[:, 6, :2]) / width), axis=1)
    steps = np.linalg.norm(np.diff(relative, axis=0), axis=2).max(axis=1)
    speed = np.r_[0, np.where(valid[1:] & valid[:-1] & np.isfinite(steps), steps, 0)]
    return speed.astype(np.float32), valid


def mine(data_dir: Path, processed_dir: Path, raw_dir: Path,
         length: int = 32, threshold: float = 0.2,
         split_names: tuple[str, ...] = ("train", "val")) -> list[dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((processed_dir / "splits.json").read_text(encoding="utf-8"))
    owner = {index: split for split in split_names for index in splits[split]}
    candidates = []
    for index in sorted(owner):
        sample = manifest[index]
        with np.load(data_dir / sample["path"]) as data:
            landmarks, mask = data["landmarks"], data["mask"]
        n = len(landmarks)
        if n < 3 * length:
            continue
        speed, valid = motion_curve(landmarks, mask)
        middle = speed[n // 3:2 * n // 3]
        middle_mean = float(middle.mean())
        if middle_mean <= 0.008:
            continue
        peak = int(np.convolve(speed, np.ones(16), mode="valid").argmax())
        if peak < length or peak + 16 > n - length:
            continue
        middle_p90 = float(np.percentile(middle, 90))
        for side, start in (("start", 0), ("end", n - length)):
            end = start + length
            segment = speed[start:end]
            pose_coverage = float(valid[start:end].mean())
            mean_ratio = float(segment.mean() / max(middle_mean, 1e-6))
            p90_ratio = float(np.percentile(segment, 90) / max(middle_p90, 1e-6))
            if pose_coverage < 0.9 or mean_ratio >= threshold or p90_ratio >= threshold:
                continue
            candidates.append({
                "manifest_index": index, "split": owner[index],
                "origin": sample["origin"], "signer": sample["signer"],
                "source_label": sample["label"], "source_npz": sample["path"],
                "source_video": str(raw_video(sample, raw_dir)),
                "edge": side, "start_frame": start, "end_frame_exclusive": end,
                "source_frames": n, "peak_start_frame": peak,
                "mean_motion_ratio": mean_ratio, "p90_motion_ratio": p90_ratio,
                "pose_coverage": pose_coverage,
                "hand_any_coverage": float(mask[start:end, :2].any(axis=1).mean()),
                "review_decision": "pending", "reviewer": "", "review_notes": "",
            })
    return candidates


def write_previews(candidates: list[dict], output_dir: Path, count: int):
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    # Include both origins, splits and boundaries where possible.
    selected = sorted(candidates, key=lambda row: (row["mean_motion_ratio"], row["p90_motion_ratio"]))
    picked = []
    for group in ("novas", "minds"):
        for row in selected:
            if row["origin"] == group and row not in picked:
                picked.append(row)
                if len(picked) >= count or len([p for p in picked if p["origin"] == group]) >= count // 2:
                    break
    for row in selected:
        if len(picked) >= count:
            break
        if row not in picked:
            picked.append(row)
    for row in picked:
        video = Path(row["source_video"])
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
        frames = []
        for frame_index in np.linspace(row["start_frame"], row["end_frame_exclusive"] - 1, 4, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (240, max(1, int(240 * h / w))))
                cv2.putText(frame, str(frame_index), (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)
                frames.append(frame)
        cap.release()
        if not frames:
            continue
        target_h = min(frame.shape[0] for frame in frames)
        strip = np.concatenate([cv2.resize(frame, (240, target_h)) for frame in frames], axis=1)
        name = f"{row['manifest_index']:03d}_{row['edge']}_{row['source_label']}.jpg"
        cv2.imwrite(str(output_dir / name), strip)


def write_contact_sheets(candidates: list[dict], preview_dir: Path, output_dir: Path):
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    for page in range((len(candidates) + 7) // 8):
        canvas = np.full((8 * 145, 800, 3), 255, dtype=np.uint8)
        for position, row in enumerate(candidates[page * 8:(page + 1) * 8]):
            name = f"{int(row['manifest_index']):03d}_{row['edge']}_{row['source_label']}.jpg"
            frame = cv2.imread(str(preview_dir / name))
            if frame is None:
                continue
            y = position * 145
            canvas[y + 25:y + 145] = cv2.resize(frame, (800, 120))
            title = f"{page * 8 + position}: {row['split']} {row['origin']} {row['source_label']} {row['edge']}"
            cv2.putText(canvas, title, (5, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(str(output_dir / f"page_{page + 1:02d}.jpg"), canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/landmarks/mixed_25"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/mixed_25"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"),
                        default=["train", "val"])
    parser.add_argument("--output", type=Path, default=Path("reports/idle_edges25/candidates.csv"))
    parser.add_argument("--preview-dir", type=Path, default=Path("reports/idle_edges25/previews"))
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--contact-dir", type=Path, default=Path("reports/idle_edges25/contact_sheets"))
    args = parser.parse_args()
    candidates = mine(args.data_dir, args.processed_dir, args.raw_dir,
                      args.length, args.threshold, tuple(args.splits))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]) if candidates else [
            "manifest_index", "split", "source_video", "start_frame", "end_frame_exclusive",
            "review_decision", "reviewer", "review_notes"])
        writer.writeheader()
        writer.writerows(candidates)
    if args.preview_count:
        write_previews(candidates, args.preview_dir, args.preview_count)
        if args.preview_count >= len(candidates):
            write_contact_sheets(candidates, args.preview_dir, args.contact_dir)
    from collections import Counter
    print(json.dumps({"candidates": len(candidates), "by_split": dict(Counter(
        row["split"] for row in candidates)), "by_origin": dict(Counter(
        row["origin"] for row in candidates)), "length": args.length,
        "threshold": args.threshold, "csv": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
