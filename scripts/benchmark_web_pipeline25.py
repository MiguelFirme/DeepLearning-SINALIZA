#!/usr/bin/env python3
"""Recorded-video benchmark through JPEG decoding, MediaPipe and frozen fusion."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path,
                        default=Path("data/raw/minds_libras/08BanheiroSinalizador11-1.mp4"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/fusion25_frozen.json"))
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("reports/fusion25_web_pipeline_cpu.json"))
    args = parser.parse_args()
    os.environ["SINALIZA_FUSION_MANIFEST_PATH"] = str(args.manifest)
    from backend.services.prediction import prediction_service
    from backend.services.video_stream import VideoStreamSession

    prediction_service.load_model()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    frame_ms, inference_ms, predictions = [], [], []
    try:
        with VideoStreamSession(prediction_service) as session:
            for _ in range(args.max_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (640, 480))
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if not ok:
                    continue
                start = time.perf_counter()
                result = session.process_jpeg(jpeg.tobytes())
                frame_ms.append((time.perf_counter() - start) * 1000)
                if result is not None:
                    inference_ms.append(result.get("processing_time_ms", 0.0))
                    predictions.append({"status": result.get("status"), "sign": result.get("sign")})
    finally:
        cap.release()
    report = {"video": str(args.video), "device": "cpu", "frames": len(frame_ms),
              "scope": "recorded 640x480 JPEG encode/decode + MediaPipe + backend fusion; excludes browser/network",
              "frame_p50_ms": float(np.percentile(frame_ms, 50)),
              "frame_p95_ms": float(np.percentile(frame_ms, 95)),
              "prediction_calls": len(inference_ms),
              "prediction_p50_ms": float(np.percentile(inference_ms, 50)) if inference_ms else None,
              "predictions": predictions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
