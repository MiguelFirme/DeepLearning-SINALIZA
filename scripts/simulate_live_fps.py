#!/usr/bin/env python3
"""Simula 6/9/15/30 FPS pelo mesmo buffer e preditor da sessão v2 do backend."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import settings
from backend.services.prediction import PredictionService
from backend.services.video_stream_v2 import TemporalWindowSessionV2
from ml.data.dataset_v2 import read_metadata
from ml.features.landmarks import LandmarkExtractor

logger = logging.getLogger(__name__)


def npz_frames(path, fps):
    """Descarta amostras; conserva os horários reais das amostras restantes."""
    with np.load(path, allow_pickle=True) as data:
        meta = read_metadata(data)
        try:
            source_fps = float(meta.get("fps", 30))
        except (ValueError, TypeError):
            source_fps = 0
        if not np.isfinite(source_fps) or not 10 <= source_fps <= 60:
            logger.warning("FPS original inválido; assumindo 30 quando não há timestamps")
            source_fps = 30
        raw, mask = data["landmarks"], data["mask"]
        times = data["timestamps"] if "timestamps" in data else np.arange(len(raw)) / source_fps
        if len(times) == 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Timestamps de entrada inválidos")
        width, height = meta.get("width"), meta.get("height")
        if not width or not height:
            raise ValueError("Simulação exige dimensões; complete os metadados sem inventar o aspecto")
        targets = np.arange(times[0], times[-1] + 1e-8, 1 / fps)
        indices = np.unique(np.clip(np.searchsorted(times, targets), 0, len(times) - 1))
        for i in indices:
            yield raw[i], mask[i], float(times[i] - times[0]), int(width), int(height)


def video_frames(path, fps):
    """Reexecuta Holistic somente nos quadros que sobreviveram ao descarte."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir {path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or not 10 <= source_fps <= 60:
        source_fps = 30
        logger.warning("FPS do vídeo inválido; assumindo 30")
    next_time, index = 0., 0
    try:
        with LandmarkExtractor() as extractor:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = index / source_fps
                index += 1
                if timestamp + 1e-8 < next_time:
                    continue
                next_time += 1 / fps
                raw, mask = extractor.extract_frame(frame)
                yield raw, mask, timestamp, frame.shape[1], frame.shape[0]
    finally:
        cap.release()


def simulate_frames(service, frames):
    """Usa relógio virtual dos quadros, inclusive no cooldown, sem esperar em sleep."""
    session = TemporalWindowSessionV2(service)
    windows, timestamps = [], []
    for raw, mask, timestamp, width, height in frames:
        previous = session.last_prediction
        result = session.process_landmarks(raw, mask, timestamp, width, height)
        timestamps.append(timestamp)
        if previous == session.last_prediction:
            continue
        if result is None:
            windows.append({"end_seconds": timestamp, "status": "cooldown", "top1": None, "confidence": None})
            continue
        top1, confidence = result.get("sign"), result.get("confidence", 0.)
        probabilities = result.get("raw_probs")
        if probabilities is not None:
            best = int(np.argmax(probabilities))
            top1 = session.predictor.label_names[best]
            confidence = float(probabilities[best])
        windows.append({"end_seconds": timestamp, "status": result.get("status"), "top1": top1,
                        "confidence": confidence, "accepted_sign": result.get("sign")})
    if not timestamps:
        raise ValueError("Entrada sem quadros")
    return {"received_frames": len(timestamps),
            "effective_fps": (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.,
            "windows": windows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--fps", type=float, nargs="+", default=[6, 9, 15, 30])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not np.isfinite(fps) or fps <= 0 for fps in args.fps):
        raise ValueError("FPS de simulação deve ser positivo")
    settings.device = args.device
    service = PredictionService()
    service.load_model(str(args.checkpoint))
    if not service.is_loaded or service.pipeline_version != 2:
        raise ValueError("Informe um checkpoint v2 treinado; a simulação não converte pesos v1")
    factory = npz_frames if args.input.suffix.lower() == ".npz" else video_frames
    report = {"input": str(args.input), "checkpoint": str(args.checkpoint), "pipeline_version": 2,
              "scope": "Simulação temporal; não mede latência JPEG/rede nem acurácia sem rótulo", "rates": {}}
    for fps in args.fps:
        result = simulate_frames(service, factory(args.input, fps))
        report["rates"][str(fps)] = result
        print(json.dumps({"requested_fps": fps, **result}, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
