"""Fluxo webcam -> JPEG -> landmarks -> Transformer -> WebSocket."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.prediction import prediction_service
from backend.services.video_stream import VideoStreamSession
from ml.features.landmarks import LandmarkExtractor


def test_websocket_classifies_48_jpeg_frames(monkeypatch):
    # O teste usa um extrator determinístico para verificar o transporte e a
    # inferência sem depender da detecção variável do MediaPipe numa imagem vazia.
    monkeypatch.setattr(LandmarkExtractor, "__enter__", lambda self: self)
    monkeypatch.setattr(LandmarkExtractor, "__exit__", lambda self, *args: None)

    def fake_extract(self, frame):
        assert frame.shape == (64, 64, 3)
        landmarks = np.zeros(346, dtype=np.float32)
        landmarks[126 + 5 * 4:126 + 5 * 4 + 3] = (0.4, 0.5, 0.0)
        landmarks[126 + 6 * 4:126 + 6 * 4 + 3] = (0.6, 0.5, 0.0)
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
            assert ready["data"]["overlay"] is True
            # O primeiro resultado conserva 48 quadros; os seguintes usam 12.
            assert VideoStreamSession(prediction_service).stride == 12
            for _ in range(48):
                ws.send_bytes(encoded.tobytes())
            for _ in range(48):
                ack = ws.receive_json()
                assert ack["type"] == "frame_ack"
                assert ack["data"]["frame_processing_ms"] > 0
                overlay = ack["data"]["overlay"]
                assert len(overlay["landmarks"]) == 346
                assert overlay["mask"] == [False, False, True, True]
                assert overlay["landmarks"][126 + 5 * 4:126 + 5 * 4 + 2] == pytest.approx([0.4, 0.5])
            reply = ws.receive_json()
            assert reply["type"] == "prediction"
            assert reply["data"]["status"] in {"recognized", "uncertain"}
            assert reply["data"]["processing_time_ms"] > 0
