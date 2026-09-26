"""Model registry."""
from ml.models.base import BaseSignModel
from ml.models.bilstm import BiLSTMModel
from ml.models.transformer import TransformerModel
from ml.models.tcn import TCNModel

MODEL_REGISTRY = {"bilstm": BiLSTMModel, "transformer": TransformerModel, "tcn": TCNModel}

def create_model(model_type: str, **kwargs) -> BaseSignModel:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Modelo '{model_type}' não encontrado. Disponíveis: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](**kwargs)
