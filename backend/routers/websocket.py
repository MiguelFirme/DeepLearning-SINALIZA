"""WebSocket: quadros JPEG da câmera para predições do modelo."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.prediction import prediction_service
from backend.services.video_stream import VideoStreamSession

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)


@router.websocket("/ws/predict")
async def websocket_predict(ws: WebSocket):
    await ws.accept()
    if not prediction_service.is_loaded:
        await ws.send_json({"type": "error", "data": {"message": "Modelo não carregado"}})
        await ws.close(code=1011)
        return

    try:
        with VideoStreamSession(prediction_service) as session:
            await ws.send_json({"type": "ready", "data": {
                "num_classes": session.predictor.num_classes,
                "target_frames": prediction_service.target_frames,
                "feature_mode": prediction_service.feature_mode,
            }})
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    try:
                        result = session.process_jpeg(message["bytes"])
                    except ValueError as exc:
                        await ws.send_json({"type": "error", "data": {"message": str(exc)}})
                        await ws.send_json({"type": "frame_ack", "data": {}})
                        continue
                    await ws.send_json({"type": "frame_ack", "data": {}})
                    if result is not None:
                        await ws.send_json({"type": "prediction", "data": {
                            "sign": result.get("sign"),
                            "confidence": result.get("confidence", 0.0),
                            "top_k": result.get("top_k", []),
                            "is_cooldown": result.get("is_cooldown", False),
                            "processing_time_ms": result.get("processing_time_ms", 0.0),
                            "status": result.get("status", "uncertain"),
                            "activity_score": result.get("activity_score", 0.0),
                        }})
                    continue

                try:
                    payload = json.loads(message.get("text") or "")
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "data": {"message": "JSON inválido"}})
                    continue
                if payload.get("type") == "control" and payload.get("data", {}).get("action") == "reset":
                    session.reset()
                    await ws.send_json({"type": "control", "data": {"status": "reset_ok"}})
                elif payload.get("type") == "ping":
                    await ws.send_json({"type": "pong", "data": {}})
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Erro no fluxo de vídeo")
        try:
            await ws.send_json({"type": "error", "data": {"message": "Falha ao processar vídeo"}})
            await ws.close(code=1011)
        except Exception:
            pass
