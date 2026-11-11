"""Coleta usa landmarks do backend, exige consentimento e separa pessoas."""
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.main import app
from backend.services.collection import LandmarkCollector
from ml.features.landmarks import LandmarkExtractor
from scripts.build_coleta_manifest import build_collection_manifest
from tests.test_pipeline_v2 import amostra


def test_coleta_e_manifesto_por_pessoa(tmp_path):
    raw, mask, _ = amostra(3)
    root = tmp_path / "coleta"
    for person in ("P1", "P2", "P3"):
        collector = LandmarkCollector(root, ["A"], "teste")
        with pytest.raises(ValueError, match="consentimento"):
            collector.start(person, "S1", "A", consent=False)
        with pytest.raises(ValueError):
            collector.start("../pasta", "S1", "A", consent=True)
        collector.start(person, "S1", "A", duration=1, consent=True, now=10)
        assert collector.feed(raw[0], mask[0], 10, 640, 480) is None
        reply = collector.feed(raw[1], mask[1], 11, 640, 480)
        assert reply["status"] == "salvo"
        with np.load(reply["path"], allow_pickle=True) as data:
            assert data["landmarks"].shape == (2, 346)
            np.testing.assert_allclose(data["timestamps"], [0, 1])
            assert json.loads(data["metadata"].item())["session"] == "S1"
    output, processed = tmp_path / "landmarks", tmp_path / "processed"
    report = build_collection_manifest(root, output, processed, ["P1"], ["P2"], ["P3"])
    assert report["splits"] == {"train": 1, "val": 1, "test": 1}
    with pytest.raises(ValueError, match="dois splits"):
        build_collection_manifest(root, output, processed, ["P1"], ["P1"], ["P3"])


def test_websocket_grava_os_mesmos_quadros(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "collection_dir", str(tmp_path))
    monkeypatch.setattr(LandmarkExtractor, "__enter__", lambda self: self)
    monkeypatch.setattr(LandmarkExtractor, "__exit__", lambda self, *args: None)
    raw, mask, _ = amostra(2)
    monkeypatch.setattr(LandmarkExtractor, "extract_frame", lambda self, frame: (raw[0], mask[0]))
    _, jpeg = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))
    with TestClient(app) as client:
        assert "Banheiro" in client.get("/api/model/info").json()["labels"]
        with client.websocket_connect("/api/ws/predict") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "control", "data": {"action": "record_start", "person": "P1", "session": "S1", "word": "Banheiro", "consent": True}})
            assert ws.receive_json()["data"]["status"] == "gravando"
            for _ in range(2):
                ws.send_bytes(jpeg.tobytes())
                assert ws.receive_json()["type"] == "frame_ack"
            ws.send_json({"type": "control", "data": {"action": "record_stop"}})
            result = ws.receive_json()
            assert result["type"] == "collection" and result["data"]["status"] == "salvo", result
    saved = list(tmp_path.rglob("*.npz"))
    assert len(saved) == 1
    with np.load(saved[0], allow_pickle=True) as data:
        np.testing.assert_array_equal(data["landmarks"], raw)
        assert data["mask"].shape == (2, 4)
