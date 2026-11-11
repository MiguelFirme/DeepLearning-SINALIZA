"""Estado de inferência por conexão para quadros JPEG da webcam."""
from __future__ import annotations

from collections import deque
import time

import cv2
import numpy as np

from ml.features.landmarks import LandmarkExtractor
from backend.services.prediction import PredictionService
from backend.services.collection import LandmarkCollector
from backend.config import settings


MAX_JPEG_BYTES = 1_000_000


class VideoStreamSession:
    def __init__(self, service: PredictionService):
        self.service = service
        self.extractor = LandmarkExtractor()
        self.predictor = service.new_session_predictor()
        self.landmarks: deque[np.ndarray] = deque(maxlen=service.target_frames)
        self.masks: deque[np.ndarray] = deque(maxlen=service.target_frames)
        self.frames_since_prediction = 0
        self.last_overlay: dict | None = None
        self.last_profile: dict | None = None
        self.collector = LandmarkCollector(settings.collection_dir, service.get_info().get("labels", []), settings.version)
        self.collection_events = deque()
        self._last_received_at = float("-inf")
        # Janelas de 48 quadros continuam intactas; atualizar a cada 12 quadros.
        self.stride = max(1, service.target_frames // 4)
        self.temporal_v2 = None
        if getattr(service, "pipeline_version", 1) == 2:
            from backend.services.video_stream_v2 import TemporalWindowSessionV2
            self.temporal_v2 = TemporalWindowSessionV2(service)

    def __enter__(self):
        self.extractor.__enter__()
        return self

    def __exit__(self, *args):
        self.collector.cancel()
        self.extractor.__exit__(*args)

    def reset(self):
        if self.temporal_v2 is not None:
            self.temporal_v2.reset()
        self.landmarks.clear()
        self.masks.clear()
        self.frames_since_prediction = 0
        self.last_overlay = None
        self.last_profile = None
        self.predictor.reset_state()

    def process_jpeg(self, jpeg: bytes, profile: bool = False) -> dict | None:
        """Extrai um quadro; classifica a cada meia janela após 48 quadros."""
        self.last_profile = None
        received_at = time.monotonic()
        # No Windows, dois quadros muito rápidos podem cair no mesmo tick do
        # relógio. Preserva a ordem com o menor incremento representável.
        received_at = max(received_at, np.nextafter(self._last_received_at, np.inf))
        self._last_received_at = received_at
        started = time.perf_counter() if profile else 0.0
        if not jpeg or len(jpeg) > MAX_JPEG_BYTES:
            raise ValueError("Quadro JPEG vazio ou maior que 1 MB")
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Não foi possível decodificar o quadro JPEG")
        if frame.shape[0] > 1080 or frame.shape[1] > 1920:
            raise ValueError("Resolução máxima: 1920x1080")

        decoded = time.perf_counter() if profile else 0.0
        landmarks, mask = self.extractor.extract_frame(frame)
        extracted = time.perf_counter() if profile else 0.0
        # Coordenadas brutas da mesma extracao usada pelo classificador.
        self.last_overlay = {"landmarks": landmarks.tolist(), "mask": mask.tolist()}
        event = self.collector.feed(landmarks, mask, received_at, frame.shape[1], frame.shape[0])
        if event:
            self.collection_events.append(event)
        if self.temporal_v2 is not None:
            result = self.temporal_v2.process_landmarks(landmarks, mask, received_at, frame.shape[1], frame.shape[0])
            if profile:
                self.last_profile = {"decode_ms": (decoded - started) * 1000,
                                     "mediapipe_ms": (extracted - decoded) * 1000,
                                     "prepare_ms": 0.,
                                     "inference_ms": (time.perf_counter() - extracted) * 1000,
                                     "window_frames": len(self.temporal_v2.buffer),
                                     "prediction_ran": result is not None}
            return result
        self.landmarks.append(landmarks)
        self.masks.append(mask)
        self.frames_since_prediction += 1
        if profile:
            self.last_profile = {
                "decode_ms": (decoded - started) * 1000,
                "mediapipe_ms": (extracted - decoded) * 1000,
                "prepare_ms": (time.perf_counter() - extracted) * 1000,
                "inference_ms": 0.0,
                "window_frames": len(self.landmarks),
                "prediction_ran": False,
            }
        if len(self.landmarks) < self.service.target_frames:
            return None
        if self.frames_since_prediction < self.stride and self.frames_since_prediction != self.service.target_frames:
            return None
        self.frames_since_prediction = 0

        part_mask = np.stack(self.masks)
        # Sem tronco e face detectáveis, o classificador sempre escolheria uma
        # das classes conhecidas, mesmo com a câmera vazia.
        if np.mean(part_mask[:, 2] | part_mask[:, 3]) < 0.4:
            self.predictor.reset_state()
            if profile and self.last_profile is not None:
                self.last_profile["prepare_ms"] = (time.perf_counter() - extracted) * 1000
            return {"sign": None, "confidence": 0.0, "top_k": [], "is_cooldown": False,
                    "processing_time_ms": 0.0, "status": "poor_tracking"}

        sequence = np.stack(self.landmarks)
        inference_started = time.perf_counter() if profile else 0.0
        result = self.service.predict(sequence, part_mask, predictor=self.predictor)
        if profile and self.last_profile is not None:
            self.last_profile["prepare_ms"] = (inference_started - extracted) * 1000
            self.last_profile["inference_ms"] = (time.perf_counter() - inference_started) * 1000
            self.last_profile["prediction_ran"] = True
        if result.get("is_cooldown"):
            return None
        result["status"] = "recognized" if result.get("sign") else "uncertain"
        if not result.get("sign"):
            result["top_k"] = []
        return result
