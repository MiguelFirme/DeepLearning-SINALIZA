"""Testes para a API FastAPI.

Nota: estes testes usam mocks para não depender de modelo treinado.
"""
import pytest
from unittest.mock import patch, MagicMock

# Mock do serviço de predição ANTES de importar o app
_mock_service = MagicMock()
_mock_service.is_loaded = True
_mock_service.get_info.return_value = {
    "type": "BiLSTMModel",
    "model_type": "bilstm",
    "num_classes": 100,
    "device": "cpu",
    "parameters": 500000,
    "smoothing": "ema",
}
_mock_service.predict.return_value = {
    "sign": "Oi",
    "confidence": 0.95,
    "top_k": [
        {"class_id": 0, "sign": "Oi", "confidence": 0.95},
        {"class_id": 1, "sign": "Bom dia", "confidence": 0.03},
    ],
    "is_cooldown": False,
    "processing_time_ms": 12.5,
}


@pytest.fixture
def client():
    """TestClient com mock do PredictionService."""
    with patch("backend.main.prediction_service", _mock_service):
        with patch("backend.routers.health.prediction_service", _mock_service), \
             patch("backend.routers.predict.prediction_service", _mock_service):
            from fastapi.testclient import TestClient
            from backend.main import app

            # Inject mock
            app.state.prediction_service = _mock_service
            yield TestClient(app)


def test_health_endpoint(client):
    """GET /health deve retornar 200."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_model_info(client):
    """GET /model/info deve retornar info do modelo."""
    resp = client.get("/api/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_type" in data


def test_dictionary_endpoint(client):
    """GET /dictionary deve retornar lista de sinais."""
    resp = client.get("/api/dictionary")
    assert resp.status_code == 200
    data = resp.json()
    assert "entries" in data
    assert "categories" in data


def test_dictionary_search(client):
    """GET /dictionary?search=oi deve funcionar."""
    resp = client.get("/api/dictionary?search=oi")
    assert resp.status_code == 200


def test_predict_reset(client):
    """POST /predict/reset deve retornar 200."""
    resp = client.post("/api/predict/reset")
    assert resp.status_code == 200
