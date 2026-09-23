"""WebSocket para inferência em tempo real."""
from __future__ import annotations
import json
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.prediction import prediction_service

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)

active_connections: set[WebSocket] = set()


@router.websocket("/ws/predict")
async def websocket_predict(ws: WebSocket):
    """
    WebSocket para predição em tempo real.

    O cliente envia frames de landmarks como JSON:
        {"type": "landmarks", "data": {"landmarks": [[...], ...], "mask": [...]}}
        {"type": "control", "data": {"action": "reset"}}

    O servidor responde:
        {"type": "prediction", "data": {"sign": ..., "confidence": ..., "top_k": [...]}}
    """
    await ws.accept()
    active_connections.add(ws)
    logger.info(f"WebSocket conectado. Ativas: {len(active_connections)}")

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "data": {"message": "JSON inválido"}})
                continue

            msg_type = msg.get("type", "")
            data = msg.get("data", {})

            if msg_type == "landmarks":
                if not prediction_service.is_loaded:
                    await ws.send_json({"type": "error", "data": {"message": "Modelo não carregado"}})
                    continue

                landmarks = np.array(data.get("landmarks", []), dtype=np.float32)
                mask_data = data.get("mask")
                mask = np.array(mask_data, dtype=bool) if mask_data else None

                if landmarks.ndim != 2 or landmarks.shape[0] == 0:
                    await ws.send_json({"type": "error", "data": {"message": "Formato de landmarks inválido"}})
                    continue

                result = prediction_service.predict(landmarks, mask)
                await ws.send_json({
                    "type": "prediction",
                    "data": {
                        "sign": result.get("sign"),
                        "confidence": result.get("confidence", 0.0),
                        "top_k": result.get("top_k", []),
                        "is_cooldown": result.get("is_cooldown", False),
                        "processing_time_ms": result.get("processing_time_ms", 0.0),
                    },
                })

            elif msg_type == "control":
                action = data.get("action", "")
                if action == "reset":
                    prediction_service.reset()
                    await ws.send_json({"type": "control", "data": {"status": "reset_ok"}})

            elif msg_type == "ping":
                await ws.send_json({"type": "pong", "data": {}})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Erro no WebSocket: {e}")
    finally:
        active_connections.discard(ws)
        logger.info(f"WebSocket desconectado. Ativas: {len(active_connections)}")
