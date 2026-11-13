"""O padrão v2 usa os pesos reais e classifica por tempo no WebSocket."""
import time
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.config import Settings, settings
from backend.main import app
import backend.services.video_stream as stream_module
from ml.features.landmarks import LandmarkExtractor


def test_websocket_default_v2_aguarda_janela_em_segundos(monkeypatch):
    defaults = Settings(_env_file=None)
    monkeypatch.setattr(settings, "fusion_manifest_path", None)
    monkeypatch.setattr(settings, "model_path", defaults.model_path)
    clock = iter(1000 + np.arange(49) / 15)
    monkeypatch.setattr(stream_module, "time", SimpleNamespace(monotonic=lambda: float(next(clock)), perf_counter=time.perf_counter))
    monkeypatch.setattr(LandmarkExtractor, "__enter__", lambda self: self)
    monkeypatch.setattr(LandmarkExtractor, "__exit__", lambda self, *args: None)
    raw = np.zeros(346, np.float32)
    raw[146:149] = [.4, .5, 0]
    raw[150:153] = [.6, .5, 0]
    mask = np.array([False, False, True, True])
    monkeypatch.setattr(LandmarkExtractor, "extract_frame", lambda self, frame: (raw, mask))
    _, jpeg = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
    with TestClient(app) as client:
        with client.websocket_connect("/api/ws/predict") as ws:
            ready = ws.receive_json()
            assert ready["data"]["num_classes"] == 23 and ready["data"]["feature_mode"] == "full"
            for index in range(1, 50):
                ws.send_bytes(jpeg.tobytes())
                ack = ws.receive_json()
                assert ack["type"] == "frame_ack" and ack["data"]["frame_id"] == index
            reply = ws.receive_json()
            assert reply["type"] == "prediction"
            assert reply["data"]["status"] in {"recognized", "uncertain"}
            assert reply["data"]["processing_time_ms"] > 0
            assert all(item["sign"] not in {"Cinco", "Espelho"} for item in reply["data"]["top_k"])
