#!/usr/bin/env python3
"""Measure frozen single/full and two-model inference on recorded landmark windows."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.fusion import FusionBundle


def summary(values: list[float]) -> dict:
    return {"p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "mean_ms": float(statistics.mean(values))}


def benchmark(manifest_path: Path, device: str, repetitions: int) -> dict:
    import torch

    torch.set_num_threads(1)
    bundle = FusionBundle(manifest_path, device)
    samples = json.loads(Path(bundle.spec["data_manifest_path"]).read_text(encoding="utf-8"))
    splits = json.loads(Path(bundle.spec["split_path"]).read_text(encoding="utf-8"))
    data_dir = Path(bundle.spec["data_manifest_path"]).parent
    # Use the same fixed validation window for both variants. No test data used.
    path = data_dir / samples[splits["val"][0]]["path"]
    with np.load(path) as data:
        landmarks, mask = data["landmarks"], data["mask"]
    records = {key: [] for key in ("full_preprocess_ms", "full_inference_ms",
                                     "pose_face_preprocess_ms", "pose_face_inference_ms",
                                     "fusion_total_ms", "full_total_ms")}
    for index in range(repetitions + 3):
        started = time.perf_counter()
        probs, timings = bundle.probabilities(landmarks, mask)
        bundle.score(probs)
        total = (time.perf_counter() - started) * 1000
        if index < 3:
            continue
        for key in timings:
            records[key].append(timings[key])
        records["fusion_total_ms"].append(total)
        records["full_total_ms"].append(timings["full_preprocess_ms"] + timings["full_inference_ms"])
    return {"device": device, "samples": repetitions, "input": str(path),
            "scope": "landmarks preprocessing + model inference; excludes JPEG, MediaPipe and transport",
            "latency": {key: summary(values) for key, values in records.items()},
            "fusion_manifest_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/fusion25_frozen.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("reports/fusion25_latency_cpu.json"))
    args = parser.parse_args()
    report = benchmark(args.manifest, args.device, args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
