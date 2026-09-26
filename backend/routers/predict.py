"""Endpoint REST de predição."""
from fastapi import APIRouter, HTTPException
import numpy as np

from backend.schemas.models import PredictRequest, PredictResponse, PredictionItem
from backend.services.prediction import prediction_service

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Prediz sinal a partir de sequência de landmarks."""
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    try:
        landmarks = np.array(request.landmarks, dtype=np.float32)
        if landmarks.ndim != 2:
            raise HTTPException(status_code=400, detail="landmarks deve ser array 2D (T, D).")

        mask = np.array(request.mask, dtype=bool) if request.mask else None

        result = prediction_service.predict(landmarks, mask)

        top_k = [PredictionItem(**item) for item in result.get("top_k", [])]

        return PredictResponse(
            sign=result.get("sign"),
            confidence=result.get("confidence", 0.0),
            top_k=top_k,
            is_cooldown=result.get("is_cooldown", False),
            processing_time_ms=result.get("processing_time_ms", 0.0),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@router.post("/predict/reset")
async def reset_state():
    """Reseta estado temporal do preditor."""
    prediction_service.reset()
    return {"status": "ok", "message": "Estado temporal resetado."}
