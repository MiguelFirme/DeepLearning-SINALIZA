"""Testes para o módulo de inferência."""
import numpy as np
import torch
import pytest

from ml.inference.predictor import SignPredictor, PredictorConfig


class DummyModel(torch.nn.Module):
    """Modelo fake que retorna logits determinísticos."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(346, num_classes)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        return self.linear(x.mean(dim=1))


@pytest.fixture
def label_map():
    return {str(i): f"sinal_{i}" for i in range(10)}


@pytest.fixture
def predictor(label_map):
    model = DummyModel(num_classes=10)
    predictor = SignPredictor(PredictorConfig(
        device="cpu",
        confidence_threshold=0.1,
        top_k=3,
        smoothing_type="none",
        cooldown_seconds=0.0,
    ))
    predictor.model = model
    predictor.num_classes = 10
    predictor.load_label_names(label_map)
    return predictor


def test_predict_returns_dict(predictor):
    """predict() deve retornar dicionário com campos obrigatórios."""
    seq = np.random.randn(48, 346).astype(np.float32)
    result = predictor.predict(seq)
    assert "sign" in result
    assert "confidence" in result
    assert "top_k" in result


def test_predict_top_k_length(predictor):
    """top_k deve ter no máximo k elementos."""
    seq = np.random.randn(48, 346).astype(np.float32)
    result = predictor.predict(seq)
    assert len(result["top_k"]) <= 3


def test_predict_confidence_range(predictor):
    """confidence deve estar entre 0 e 1."""
    seq = np.random.randn(48, 346).astype(np.float32)
    result = predictor.predict(seq)
    assert 0.0 <= result["confidence"] <= 1.0


def test_reset_clears_state(predictor):
    """reset() deve limpar histórico de smoothing."""
    seq = np.random.randn(48, 346).astype(np.float32)
    predictor.predict(seq)
    predictor.reset_state()
    # Não deve lançar exceção
    result = predictor.predict(seq)
    assert result is not None
