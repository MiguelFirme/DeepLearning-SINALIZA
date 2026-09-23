"""Configuração do backend."""
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Sinaliza API"
    version: str = "1.0.0"
    debug: bool = False

    # Modelo
    model_path: str = "artifacts/sinaliza_model.pt"
    device: str = "cpu"
    confidence_threshold: float = 0.3
    top_k: int = 5
    smoothing_type: str = "ema"
    ema_alpha: float = 0.6
    temperature: float = 1.0

    # Sequência
    target_frames: int = 48

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

    # WebSocket
    ws_max_connections: int = 10

    # Paths
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"

    class Config:
        env_prefix = "SINALIZA_"
        env_file = ".env"


settings = Settings()
