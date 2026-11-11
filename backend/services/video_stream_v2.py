"""Sessão temporal reutilizada pela webcam e pela simulação offline v2."""
from __future__ import annotations

from collections import deque

import numpy as np

from ml.features.temporal_v2 import resample_by_time


class TemporalWindowSessionV2:
    def __init__(self, service):
        self.service = service
        self.cfg = service.pipeline_cfg
        self.predictor = service.new_session_predictor()
        self.buffer = deque()
        self.last_prediction = None
        self.dimensions = None
        self.predictor.clock = lambda: self.last_prediction if self.last_prediction is not None else 0.

    def reset(self):
        self.buffer.clear()
        self.last_prediction = None
        self.dimensions = None
        self.predictor.reset_state()

    def process_landmarks(self, raw, mask, timestamp, width, height):
        """Recebe um quadro bruto e usa exatamente a janela temporal do backend."""
        if self.dimensions is not None and self.dimensions != (width, height):
            self.reset()
        self.dimensions = (width, height)
        if self.buffer and timestamp <= self.buffer[-1][0]:
            raise ValueError("Quadros v2 precisam ter timestamps crescentes")
        self.buffer.append((float(timestamp), np.asarray(raw).copy(), np.asarray(mask).copy()))
        horizon = max(6.0, self.cfg.window_seconds * 2)
        while len(self.buffer) > 2 and self.buffer[1][0] < timestamp - horizon:
            self.buffer.popleft()
        if timestamp - self.buffer[0][0] < self.cfg.window_seconds:
            return None
        if self.last_prediction is not None and timestamp - self.last_prediction < .4 - 1e-8:
            return None
        self.last_prediction = timestamp
        times, raw, mask = zip(*self.buffer)
        raw, mask, times = np.asarray(raw), np.asarray(mask), np.asarray(times)
        _, sampled_mask, _ = resample_by_time(raw, mask, times, timestamp,
                                             self.cfg.target_fps, self.cfg.n_samples)
        if np.mean(sampled_mask[:, 2] | sampled_mask[:, 3]) < .4:
            self.predictor.reset_state()
            return {"sign": None, "confidence": 0., "top_k": [], "status": "poor_tracking",
                    "processing_time_ms": 0., "is_cooldown": False}
        result = self.service.predict(raw, mask, predictor=self.predictor, timestamps=times,
                                      width=width, height=height)
        if result.get("is_cooldown"):
            return None
        result.setdefault("status", "recognized" if result.get("sign") else "uncertain")
        if not result.get("sign"):
            result["top_k"] = []
        return result
