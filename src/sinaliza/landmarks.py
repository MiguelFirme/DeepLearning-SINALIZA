"""Extração de landmarks corporais e faciais com MediaPipe Holistic."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


POSE_FEATURES = 33 * 4  # x, y, z e visibilidade
FACE_FEATURES = 468 * 3  # x, y e z
HAND_FEATURES = 21 * 3  # x, y e z para cada mão
FEATURE_DIM = POSE_FEATURES + FACE_FEATURES + 2 * HAND_FEATURES


def _flatten(landmarks: object | None, count: int, fields: tuple[str, ...]) -> np.ndarray:
    if landmarks is None:
        return np.zeros(count * len(fields), dtype=np.float32)

    points = getattr(landmarks, "landmark")
    values = [getattr(point, field) for point in points for field in fields]
    vector = np.asarray(values, dtype=np.float32)
    if vector.size != count * len(fields):
        raise ValueError("O MediaPipe retornou uma quantidade inesperada de landmarks")
    return vector


def resultado_para_vetor(results: object) -> np.ndarray:
    """Converte o resultado do MediaPipe em um único vetor numérico."""
    return np.concatenate(
        (
            _flatten(results.pose_landmarks, 33, ("x", "y", "z", "visibility")),
            _flatten(results.face_landmarks, 468, ("x", "y", "z")),
            _flatten(results.left_hand_landmarks, 21, ("x", "y", "z")),
            _flatten(results.right_hand_landmarks, 21, ("x", "y", "z")),
        )
    )


def extrair_video(video_path: Path) -> np.ndarray:
    """Extrai uma matriz no formato (frames, features) de um arquivo de vídeo."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

    frames: list[np.ndarray] = []
    holistic_api = mp.solutions.holistic
    try:
        with holistic_api.Holistic(
            static_image_mode=False,
            model_complexity=1,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                frames.append(resultado_para_vetor(holistic.process(rgb_frame)))
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"O vídeo não contém frames legíveis: {video_path}")
    return np.stack(frames).astype(np.float32)
