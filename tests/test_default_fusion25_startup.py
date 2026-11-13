"""Inicialização com checkpoint v2 real e compatibilidade explícita com v1."""
from pathlib import Path

from fastapi.testclient import TestClient

from backend.config import Settings, settings
from backend.main import app
from backend.services.prediction import prediction_service


ROOT = Path(__file__).resolve().parents[1]


def test_default_startup_uses_original_25_class_fusion(monkeypatch):
    manifest = "configs/fusion25_frozen.json"
    assert (ROOT / manifest).is_file()
    monkeypatch.setattr(settings, "fusion_manifest_path", manifest)
    with TestClient(app) as client:
        health = client.get("/api/health")
        info = client.get("/api/model/info")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        assert health.json()["num_classes"] == 25
        assert info.status_code == 200
        assert info.json()["model_type"] == "transformer_fusion"
        assert info.json()["num_classes"] == 25
        assert prediction_service._fusion.feature_modes == {"pose_face": "pose_face", "full": "full"}
        assert prediction_service._fusion.model_types == {"pose_face": "transformer", "full": "transformer"}
        assert prediction_service._fusion.spec["fusion"] == {"first": "pose_face", "top_k": 5, "pose_weight": 0.22}


def test_default_startup_uses_trained_transformer_v2(monkeypatch):
    monkeypatch.delenv("SINALIZA_FUSION_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("SINALIZA_MODEL_PATH", raising=False)
    defaults = Settings(_env_file=None)
    assert defaults.fusion_manifest_path is None
    assert (ROOT / defaults.model_path).is_file()
    monkeypatch.setattr(settings, "fusion_manifest_path", defaults.fusion_manifest_path)
    monkeypatch.setattr(settings, "model_path", defaults.model_path)
    with TestClient(app) as client:
        health, info = client.get("/api/health").json(), client.get("/api/model/info").json()
        assert health["model_loaded"] is True and health["num_classes"] == 23
        assert info["model_type"] == "transformer_v2" and info["pipeline_version"] == 2
        assert info["feature_dim"] == 725 and info["target_fps"] == 15 and info["window_seconds"] == 3.2
        assert len(info["labels"]) == 23
        assert "Água" in info["labels"] and "Cinco" not in info["labels"] and "Espelho" not in info["labels"]
        assert prediction_service._fusion is None
