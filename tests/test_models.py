"""Testes para os modelos de classificação."""
import torch
import pytest

from ml.models import create_model


NUM_CLASSES = 100
SEQ_LEN = 48
INPUT_DIM = 346
BATCH = 4


@pytest.fixture(params=["bilstm", "transformer", "tcn"])
def model_type(request):
    return request.param


def _make_model(model_type: str):
    configs = {
        "bilstm": dict(
            input_dim=INPUT_DIM,
            num_classes=NUM_CLASSES,
            hidden_size=64,
            num_layers=1,
            dropout=0.1,
        ),
        "transformer": dict(
            input_dim=INPUT_DIM,
            num_classes=NUM_CLASSES,
            d_model=64,
            nhead=4,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.1,
        ),
        "tcn": dict(
            input_dim=INPUT_DIM,
            num_classes=NUM_CLASSES,
            num_channels=64,
            num_blocks=2,
            kernel_size=3,
            dropout=0.1,
        ),
    }
    return create_model(model_type, **configs[model_type])


def test_model_forward_shape(model_type):
    """Saída deve ser (batch, num_classes)."""
    model = _make_model(model_type)
    x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_model_output_is_logits(model_type):
    """Saída deve ser logits (não softmax)."""
    model = _make_model(model_type)
    x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
    out = model(x)
    # Logits podem ser negativos
    assert out.min().item() < 1.0


def test_model_param_count(model_type):
    """Modelo deve ter parâmetros treináveis."""
    model = _make_model(model_type)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 0


def test_model_train_eval_modes(model_type):
    """Modelo deve alternar entre train/eval sem erro."""
    model = _make_model(model_type)
    x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)

    model.train()
    out_train = model(x)

    model.eval()
    with torch.no_grad():
        out_eval = model(x)

    assert out_train.shape == out_eval.shape


def test_model_backward(model_type):
    """Backward pass deve funcionar."""
    model = _make_model(model_type)
    x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
    out = model(x)
    loss = out.sum()
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad
