"""Configuração do backend."""
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Sinaliza API"
    version: str = "1.0.0"
    debug: bool = False

    # Modelo
    model_path: str = "Modelos salvos/transformer_v2_grupo39_full/best.pt"
    fusion_manifest_path: str | None = None
    device: str = "cpu"
    confidence_threshold: float = 0.3
    top_k: int = 5
    smoothing_type: str = "ema"
    ema_alpha: float = 0.6
    temperature: float = 1.0
    perf_logging: bool = False
    perf_detail: bool = False

    # Sequência
    target_frames: int = 48

    # IA (montagem de frases, Google Gemini). Sem chave, o backend usa um fallback simples.
    gemini_api_key: str | None = None
    llm_model: str = "gemini-3.1-flash-lite"
    llm_timeout_s: float = 30.0
    llm_max_words: int = 20

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

    # WebSocket
    ws_max_connections: int = 10

    # Paths
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    collection_dir: str = "data/coleta"

    class Config:
        env_prefix = "SINALIZA_"
        env_file = ".env"


settings = Settings()
