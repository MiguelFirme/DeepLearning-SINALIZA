"""Fluxo webcam -> JPEG -> landmarks -> Transformer -> WebSocket."""
from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from ml.features.landmarks import LandmarkExtractor


def test_websocket_classifies_48_jpeg_frames(monkeypatch):
    # O teste usa um extrator determinístico para verificar o transporte e a
    # inferência sem depender da detecção variável do MediaPipe numa imagem vazia.
    monkeypatch.setattr(LandmarkExtractor, "__enter__", lambda self: self)
    monkeypatch.setattr(LandmarkExtractor, "__exit__", lambda self, *args: None)

    state = {"moving": False, "frame": 0}

    def fake_extract(self, frame):
        assert frame.shape == (64, 64, 3)
        landmarks = np.zeros(346, dtype=np.float32)
        landmarks[126 + 5 * 4:126 + 5 * 4 + 3] = (0.4, 0.5, 0.0)
        landmarks[126 + 6 * 4:126 + 6 * 4 + 3] = (0.6, 0.5, 0.0)
        landmarks[126 + 9 * 4:126 + 9 * 4 + 3] = (
            0.4 + (0.5 * state["frame"] / 47 if state["moving"] else 0), 0.7, 0.0
        )
        state["frame"] += 1
        return landmarks, np.array([False, False, True, True])

    monkeypatch.setattr(LandmarkExtractor, "extract_frame", fake_extract)
    ok, encoded = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok

    with TestClient(app) as client:
        assert client.get("/api/health").json()["model_loaded"] is True
        with client.websocket_connect("/api/ws/predict") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["data"]["num_classes"] == 8
            assert ready["data"]["target_frames"] == 48
            for _ in range(48):
                ws.send_bytes(encoded.tobytes())
            for _ in range(48):
                assert ws.receive_json()["type"] == "frame_ack"
            reply = ws.receive_json()
            assert reply["type"] == "prediction"
            assert reply["data"]["status"] == "idle"
            assert reply["data"]["sign"] is None
            assert reply["data"]["top_k"] == []

            ws.send_json({"type": "control", "data": {"action": "reset"}})
            assert ws.receive_json()["type"] == "control"
            state.update(moving=True, frame=0)
            for _ in range(48):
                ws.send_bytes(encoded.tobytes())
            for _ in range(48):
                assert ws.receive_json()["type"] == "frame_ack"
            reply = ws.receive_json()
            assert reply["type"] == "prediction"
            assert reply["data"]["activity_score"] > 0.35
            assert reply["data"]["status"] in {"recognized", "uncertain"}
