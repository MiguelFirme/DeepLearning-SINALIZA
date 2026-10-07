#!/usr/bin/env python3
"""Envia um MP4 pelo mesmo pipeline JPEG usado pela webcam web."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.prediction import prediction_service
from backend.services.video_stream import VideoStreamSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()
    prediction_service.load_model()
    if not prediction_service.is_loaded:
        raise RuntimeError("Modelo de 8 classes não carregado")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(source_fps / args.fps))
    sampled = 0
    predictions = []
    try:
        with VideoStreamSession(prediction_service) as session:
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % step == 0:
                    frame = cv2.resize(frame, (640, 480))
                    encoded_ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
                    if not encoded_ok:
                        raise RuntimeError("Falha ao codificar JPEG")
                    result = session.process_jpeg(jpeg.tobytes())
                    sampled += 1
                    if result is not None:
                        predictions.append({
                            "status": result.get("status"),
                            "activity_score": result.get("activity_score"),
                            "sign": result["sign"],
                            "confidence": result["confidence"],
                            "top_k": result["top_k"],
                        })
                frame_index += 1
    finally:
        cap.release()
    print(json.dumps({"video": str(args.video), "source_fps": source_fps,
                      "sampled_frames": sampled, "predictions": predictions},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
