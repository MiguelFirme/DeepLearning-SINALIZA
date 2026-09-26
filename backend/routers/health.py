"""Endpoint de health check."""
from fastapi import APIRouter
from backend.config import settings
from backend.services.prediction import prediction_service
from backend.schemas.models import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    info = prediction_service.get_info()
    return HealthResponse(
        status="ok",
        model_loaded=prediction_service.is_loaded,
        device=info.get("device", settings.device),
        num_classes=info.get("num_classes", 0),
        version=settings.version,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    info = prediction_service.get_info()
    return ModelInfoResponse(
        model_type=info.get("type", "unknown"),
        num_classes=info.get("num_classes", 0),
        device=info.get("device", "cpu"),
        parameters=info.get("trainable", 0),
        smoothing=info.get("smoothing", "none"),
    )
