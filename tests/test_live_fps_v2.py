"""Robustez temporal da sessão real e do simulador com checkpoint sintético."""
from dataclasses import asdict
import json
from types import SimpleNamespace
import time

import cv2
import numpy as np
import torch

from backend.services.prediction import PredictionService
from backend.services.video_stream import VideoStreamSession
from ml.features.landmarks import LandmarkExtractor
from ml.features.pipeline_v2 import PipelineV2Config, prepare_features_v2
from ml.models import create_model
from scripts.simulate_live_fps import npz_frames, simulate_frames
from tests.test_pipeline_v2 import amostra


def test_sessao_real_e_mesmo_movimento_em_quatro_fps(tmp_path, monkeypatch):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    cfg = PipelineV2Config()
    kwargs = dict(d_model=16, nhead=4, num_layers=1, dim_feedforward=32, dropout=0., max_seq_len=48, classifier_hidden=16)
    model = create_model("transformer", input_dim=725, num_classes=2, **kwargs)
    checkpoint = {"pipeline_version": 2, "pipeline_config": asdict(cfg), "feature_dim": 725,
                  "target_fps": 15., "window_seconds": 3.2, "target_frames": 48, "feature_mode": "full",
                  "model_type": "transformer", "model_kwargs": kwargs, "num_classes": 2,
                  "label_map": {"A": 0, "B": 1}, "model_state_dict": model.state_dict()}
    path = tmp_path / "modelo.pt"
    torch.save(checkpoint, path)
    service = PredictionService()
    service.load_model(str(path))
    raw, mask, _ = amostra(241)
    times = np.arange(len(raw)) / 30
    raw[:, :126:3] += times[:, None] * .02
    data_path = tmp_path / "movimento.npz"
    np.savez(data_path, landmarks=raw, mask=mask, timestamps=times,
             metadata=json.dumps({"fps": 30, "width": 640, "height": 480}))
    velocities = []
    for fps in (6, 9, 15, 30):
        frames = list(npz_frames(data_path, fps))
        report = simulate_frames(service, frames)
        assert report["windows"] and report["windows"][0]["end_seconds"] >= 3.2
        ends = np.asarray([row["end_seconds"] for row in report["windows"]])
        assert (np.diff(ends) >= .4 - 1e-8).all()
        assert any(row["top1"] in {"A", "B"} for row in report["windows"])
        values, _, _ = prepare_features_v2(np.stack([row[0] for row in frames]), np.stack([row[1] for row in frames]),
                                           np.array([row[2] for row in frames]), 640, 480, cfg)
        velocities.append(values[1:, 428:491])
    for values in velocities[1:]:
        np.testing.assert_allclose(velocities[0], values, atol=3e-5)
    # Atravessa também o JPEG do WebSocket, com relógio injetado apenas aqui.
    import backend.services.video_stream as stream_module
    clock = iter(np.arange(75) / 9)
    monkeypatch.setattr(stream_module, "time", SimpleNamespace(monotonic=lambda: float(next(clock)), perf_counter=time.perf_counter))
    monkeypatch.setattr(LandmarkExtractor, "__enter__", lambda self: self)
    monkeypatch.setattr(LandmarkExtractor, "__exit__", lambda self, *args: None)
    monkeypatch.setattr(LandmarkExtractor, "extract_frame", lambda self, frame: (raw[0], mask[0]))
    _, jpeg = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
    with VideoStreamSession(service) as stream:
        assert stream.temporal_v2 is not None
        replies = [stream.process_jpeg(jpeg.tobytes(), profile=True) for _ in range(75)]
        assert all(reply is None for reply in replies[:29])
        assert any(reply is not None for reply in replies[29:])
