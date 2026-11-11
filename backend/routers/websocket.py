"""WebSocket: quadros JPEG da câmera para predições do modelo."""
from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import settings
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
            frame_id = 0
            perf_started = time.perf_counter()
            perf_count = 0
            perf_calls = 0
            perf_totals = {key: 0.0 for key in (
                "decode_ms", "mediapipe_ms", "prepare_ms", "inference_ms", "ack_ms"
            )}
            await ws.send_json({"type": "ready", "data": {
                "num_classes": session.predictor.num_classes,
                "target_frames": prediction_service.target_frames,
                "feature_mode": prediction_service.feature_mode,
                "overlay": True,
            }})
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    frame_id += 1
                    try:
                        frame_started = time.perf_counter()
                        result = session.process_jpeg(message["bytes"], profile=settings.perf_logging)
                    except ValueError as exc:
                        await ws.send_json({"type": "error", "data": {"message": str(exc)}})
                        await ws.send_json({"type": "frame_ack", "data": {"frame_id": frame_id}})
                        continue
                    processed_ms = (time.perf_counter() - frame_started) * 1000
                    ack_data = {
                        "frame_id": frame_id,
                        "overlay": session.last_overlay,
                        "frame_processing_ms": processed_ms,
                    }
                    if settings.perf_logging:
                        ack_data["profile"] = session.last_profile
                    ack_started = time.perf_counter()
                    await ws.send_json({"type": "frame_ack", "data": ack_data})
                    while session.collection_events:
                        await ws.send_json({"type": "collection", "data": session.collection_events.popleft()})
                    if settings.perf_logging:
                        stages = session.last_profile or {}
                        ack_ms = (time.perf_counter() - ack_started) * 1000
                        perf_count += 1
                        perf_calls += int(bool(stages.get("prediction_ran")))
                        for key in ("decode_ms", "mediapipe_ms", "prepare_ms", "inference_ms"):
                            perf_totals[key] += stages.get(key, 0.0)
                        perf_totals["ack_ms"] += ack_ms
                        if settings.perf_detail:
                            logger.info(
                                "[PERF DETALHE] quadro=%d jpeg_kb=%.1f decode=%.1fms mediapipe=%.1fms "
                                "preparo=%.1fms modelo=%.1fms ack=%.1fms total=%.1fms",
                                frame_id, len(message["bytes"]) / 1024,
                                stages.get("decode_ms", 0), stages.get("mediapipe_ms", 0),
                                stages.get("prepare_ms", 0), stages.get("inference_ms", 0),
                                ack_ms, processed_ms,
                            )
                        elapsed = time.perf_counter() - perf_started
                        if elapsed >= 3.0:
                            largest = max((
                                ("JPEG", perf_totals["decode_ms"]),
                                ("MediaPipe", perf_totals["mediapipe_ms"]),
                                ("preparo", perf_totals["prepare_ms"]),
                                ("modelo", perf_totals["inference_ms"]),
                                ("retorno", perf_totals["ack_ms"]),
                            ), key=lambda item: item[1])[0]
                            logger.info(
                                "[PERF RESUMO] %.1fs | analise=%.1f FPS | JPEG=%.0fms/quadro | "
                                "MediaPipe=%.0fms/quadro | modelo=%.0fms/predicao (%d vezes) | "
                                "retorno=%.0fms/quadro | maior_etapa=%s",
                                elapsed, perf_count / elapsed,
                                perf_totals["decode_ms"] / perf_count,
                                perf_totals["mediapipe_ms"] / perf_count,
                                perf_totals["inference_ms"] / max(1, perf_calls), perf_calls,
                                perf_totals["ack_ms"] / perf_count, largest,
                            )
                            perf_started = time.perf_counter()
                            perf_count = 0
                            perf_calls = 0
                            perf_totals = {key: 0.0 for key in perf_totals}
                    if result is not None:
                        await ws.send_json({"type": "prediction", "data": {
                            "sign": result.get("sign"),
                            "confidence": result.get("confidence", 0.0),
                            "top_k": result.get("top_k", []),
                            "is_cooldown": result.get("is_cooldown", False),
                            "processing_time_ms": result.get("processing_time_ms", 0.0),
                            "timings": result.get("timings", {}),
                            "fusion_mode": result.get("fusion_mode"),
                            "status": result.get("status", "uncertain"),
                        }})
                    continue

                try:
                    payload = json.loads(message.get("text") or "")
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "data": {"message": "JSON inválido"}})
                    continue
                action = payload.get("data", {}).get("action")
                if payload.get("type") == "control" and action in {"record_start", "record_stop", "record_cancel"}:
                    try:
                        data = payload["data"]
                        if action == "record_start":
                            reply = session.collector.start(data.get("person"), data.get("session"), data.get("word"),
                                                            data.get("duration", 4), data.get("consent", False))
                        elif action == "record_stop":
                            reply = session.collector.finish()
                        else:
                            reply = session.collector.cancel()
                    except (ValueError, TypeError) as exc:
                        reply = {"status": "erro", "message": str(exc)}
                    await ws.send_json({"type": "collection", "data": reply})
                elif payload.get("type") == "control" and action == "reset":
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
