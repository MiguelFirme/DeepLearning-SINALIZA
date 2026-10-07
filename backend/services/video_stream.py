"""Estado de inferência por conexão para quadros JPEG da webcam."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ml.features.landmarks import LandmarkExtractor
from ml.inference.activity import pose_arm_motion
from backend.services.prediction import PredictionService
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
        self.stride = max(1, service.target_frames // 2)

    def __enter__(self):
        self.extractor.__enter__()
        return self

    def __exit__(self, *args):
        self.extractor.__exit__(*args)

    def reset(self):
        self.landmarks.clear()
        self.masks.clear()
        self.frames_since_prediction = 0
        self.predictor.reset_state()

    def process_jpeg(self, jpeg: bytes) -> dict | None:
        """Extrai um quadro; classifica a cada meia janela após 48 quadros."""
        if not jpeg or len(jpeg) > MAX_JPEG_BYTES:
            raise ValueError("Quadro JPEG vazio ou maior que 1 MB")
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Não foi possível decodificar o quadro JPEG")
        if frame.shape[0] > 1080 or frame.shape[1] > 1920:
            raise ValueError("Resolução máxima: 1920x1080")

        landmarks, mask = self.extractor.extract_frame(frame)
        self.landmarks.append(landmarks)
        self.masks.append(mask)
        self.frames_since_prediction += 1
        if len(self.landmarks) < self.service.target_frames:
            return None
        if self.frames_since_prediction < self.stride and self.frames_since_prediction != self.service.target_frames:
            return None
        self.frames_since_prediction = 0

        part_mask = np.stack(self.masks)
        # Sem tronco e face detectáveis, o classificador sempre escolheria uma
        # das oito classes, mesmo com a câmera vazia.
        if np.mean(part_mask[:, 2] | part_mask[:, 3]) < 0.4:
            self.predictor.reset_state()
            return {"sign": None, "confidence": 0.0, "top_k": [], "is_cooldown": False,
                    "processing_time_ms": 0.0, "status": "poor_tracking", "activity_score": 0.0}

        window = np.stack(self.landmarks)
        activity = pose_arm_motion(window, part_mask)
        if activity < settings.min_arm_motion:
            self.predictor.reset_state()
            return {"sign": None, "confidence": 0.0, "top_k": [], "is_cooldown": False,
                    "processing_time_ms": 0.0, "status": "idle", "activity_score": activity}

        result = self.service.predict(window, part_mask, predictor=self.predictor)
        if result.get("is_cooldown"):
            return None
        result["status"] = "recognized" if result.get("sign") else "uncertain"
        result["activity_score"] = activity
        if not result.get("sign"):
            result["top_k"] = []
            self.predictor.reset_state()
        return result
