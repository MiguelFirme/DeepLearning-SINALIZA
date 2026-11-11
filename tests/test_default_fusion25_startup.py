"""The web app must load the archived ConvBiLSTM pair by default."""
from pathlib import Path

from fastapi.testclient import TestClient

from backend.config import Settings, settings
from backend.main import app
from backend.services.prediction import prediction_service


ROOT = Path(__file__).resolve().parents[1]


def test_default_startup_uses_conv_bilstm_25_class_fusion(monkeypatch):
    monkeypatch.delenv("SINALIZA_FUSION_MANIFEST_PATH", raising=False)
    manifest = Settings(_env_file=None).fusion_manifest_path
    assert manifest == "configs/fusion25_conv_bilstm_default.json"
    assert (ROOT / manifest).is_file()
    monkeypatch.setattr(settings, "fusion_manifest_path", manifest)
    with TestClient(app) as client:
        health = client.get("/api/health")
        info = client.get("/api/model/info")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        assert health.json()["num_classes"] == 25
        assert info.status_code == 200
        assert info.json()["model_type"] == "conv_bilstm_attention_fusion"
        assert info.json()["num_classes"] == 25
        assert prediction_service._fusion.feature_modes == {"pose_face": "pose_face", "full": "full"}
        assert prediction_service._fusion.model_types == {"pose_face": "conv_bilstm_attention", "full": "conv_bilstm_attention"}
        assert prediction_service._fusion.spec["fusion"] == {"first": "pose_face", "top_k": 5, "pose_weight": 0.22}
