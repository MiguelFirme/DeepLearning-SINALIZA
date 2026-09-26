"""Schemas Pydantic para a API."""
from __future__ import annotations
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    device: str
    num_classes: int
    version: str


class PredictionItem(BaseModel):
    class_id: int
    sign: str
    confidence: float


class PredictRequest(BaseModel):
    landmarks: list[list[float]] = Field(..., description="Sequência de landmarks (T, D)")
    mask: list[bool] | None = Field(None, description="Máscara de frames válidos")


class PredictResponse(BaseModel):
    sign: str | None
    confidence: float
    top_k: list[PredictionItem]
    is_cooldown: bool = False
    processing_time_ms: float = 0.0


class DictionaryEntry(BaseModel):
    sign_id: int
    name: str
    category: str = ""
    description: str = ""
    example_video_url: str | None = None


class DictionaryResponse(BaseModel):
    entries: list[DictionaryEntry]
    total: int
    categories: list[str]


class ModelInfoResponse(BaseModel):
    model_type: str
    num_classes: int
    device: str
    parameters: int = 0
    smoothing: str = "ema"


class WebSocketMessage(BaseModel):
    type: str  # landmarks | control | ping
    data: dict | None = None
