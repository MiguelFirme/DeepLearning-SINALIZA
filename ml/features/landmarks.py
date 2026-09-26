"""
Extração de landmarks via MediaPipe Holistic.

Landmarks:
    Mão esquerda/direita: 21 pontos (x,y,z) = 63 valores cada
    Pose: 25 pontos selecionados (x,y,z,vis) = 100 valores
    Face: 40 pontos selecionados (x,y,z) = 120 valores
    Total: 346 valores por frame
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

FACE_SELECTED_INDICES = [
    46,52,53,55,65, 276,282,283,285,295,  # Sobrancelhas
    33,133,160,144, 362,263,387,373,       # Olhos
    1,2,98,327,                            # Nariz
    13,14,78,308,82,312,87,317,0,267,      # Boca
    10,152,234,454,127,356,                # Contorno
    168,6,                                 # Orientação
]

POSE_SELECTED_INDICES = [
    0,2,5,7,8, 11,12,13,14,15,16,
    17,18,19,20,21,22, 23,24,25,26,27,28,29,30,
]

HAND_LANDMARKS = 21
HAND_DIMS = 3
POSE_DIMS = 4
FACE_DIMS = 3

@dataclass
class LandmarkConfig:
    use_hands: bool = True
    use_pose: bool = True
    use_face: bool = True
    face_indices: list[int] = field(default_factory=lambda: FACE_SELECTED_INDICES.copy())
    pose_indices: list[int] = field(default_factory=lambda: POSE_SELECTED_INDICES.copy())
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False

    @property
    def feature_dim(self) -> int:
        dim = 0
        if self.use_hands:
            dim += 2 * HAND_LANDMARKS * HAND_DIMS
        if self.use_pose:
            dim += len(self.pose_indices) * POSE_DIMS
        if self.use_face:
            dim += len(self.face_indices) * FACE_DIMS
        return dim


class LandmarkExtractor:
    """Extrai landmarks via MediaPipe Holistic."""

    def __init__(self, config: Optional[LandmarkConfig] = None):
        self.config = config or LandmarkConfig()
        self._holistic = None

    def __enter__(self):
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=self.config.static_image_mode,
            model_complexity=2,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        return self

    def __exit__(self, *args):
        if self._holistic:
            self._holistic.close()
            self._holistic = None

    def extract_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extrai landmarks de um frame. Retorna (vetor, mascara_deteccao)."""
        if self._holistic is None:
            raise RuntimeError("Use com context manager: with LandmarkExtractor() as ext:")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._holistic.process(rgb)
        parts, mask = [], []

        if self.config.use_hands:
            for hand_lm in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand_lm:
                    arr = np.array([[l.x,l.y,l.z] for l in hand_lm.landmark], dtype=np.float32)
                    parts.append(arr.flatten())
                    mask.append(True)
                else:
                    parts.append(np.zeros(HAND_LANDMARKS*HAND_DIMS, dtype=np.float32))
                    mask.append(False)

        if self.config.use_pose and results.pose_landmarks:
            all_pose = np.array([[l.x,l.y,l.z,l.visibility] for l in results.pose_landmarks.landmark], dtype=np.float32)
            parts.append(all_pose[self.config.pose_indices].flatten())
            mask.append(True)
        elif self.config.use_pose:
            parts.append(np.zeros(len(self.config.pose_indices)*POSE_DIMS, dtype=np.float32))
            mask.append(False)

        if self.config.use_face and results.face_landmarks:
            all_face = np.array([[l.x,l.y,l.z] for l in results.face_landmarks.landmark], dtype=np.float32)
            parts.append(all_face[self.config.face_indices].flatten())
            mask.append(True)
        elif self.config.use_face:
            parts.append(np.zeros(len(self.config.face_indices)*FACE_DIMS, dtype=np.float32))
            mask.append(False)

        return np.concatenate(parts).astype(np.float32), np.array(mask, dtype=bool)

    def extract_video(self, video_path: str | Path, max_frames: int | None = None):
        """Extrai landmarks de todos os frames de um vídeo."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Erro ao abrir: {video_path}")

        metadata = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

        frames_data, masks_data = [], []
        count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and count >= max_frames:
                    break
                vec, mask = self.extract_frame(frame)
                frames_data.append(vec)
                masks_data.append(mask)
                count += 1
        finally:
            cap.release()

        if frames_data:
            return np.stack(frames_data), np.stack(masks_data), metadata
        return np.empty((0, self.config.feature_dim)), np.empty((0,4), dtype=bool), metadata
