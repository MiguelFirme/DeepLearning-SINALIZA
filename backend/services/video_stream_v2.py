"""Sessão temporal reutilizada pela webcam e pela simulação offline v2."""
from __future__ import annotations

from collections import deque

import numpy as np

from ml.features.segmentation_v2 import centered_window_end, motion_segments
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
        self.classified_until = -np.inf   # fim do último trecho já classificado
        self.last_check = None

    def reset(self):
        self.buffer.clear()
        self.last_prediction = None
        self.dimensions = None
        self.classified_until = -np.inf
        self.last_check = None
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
        if getattr(self.cfg, "segmentation", False):
            return self._process_segmented(timestamp, width, height)
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

    def _process_segmented(self, timestamp, width, height):
        """Classifica cada sinal uma vez, com a janela centrada da validação.

        Espera o repouso que encerra o gesto e o tempo para a janela centrada
        terminar; então classifica sem EMA nem cooldown herdados de janelas
        anteriores (que misturavam repouso e sinal parcial).
        """
        if self.last_check is not None and timestamp - self.last_check < .1:
            return None
        self.last_check = timestamp
        times, raw, mask = zip(*self.buffer)
        raw, mask, times = np.asarray(raw), np.asarray(mask), np.asarray(times)
        pending = [seg for seg in motion_segments(raw, mask, times, width, height)
                   if seg[1] > self.classified_until]
        if not pending:
            return None
        segment = pending[0]
        end_time = centered_window_end(segment, self.cfg.window_seconds)
        if end_time > timestamp:
            return None          # a janela centrada ainda não terminou
        self.classified_until = segment[1]
        if end_time - self.cfg.window_seconds < times[0]:
            return None          # o início do sinal já saiu do buffer
        _, sampled_mask, _ = resample_by_time(raw, mask, times, end_time,
                                             self.cfg.target_fps, self.cfg.n_samples)
        if np.mean(sampled_mask[:, 2] | sampled_mask[:, 3]) < .4:
            return {"sign": None, "confidence": 0., "top_k": [], "status": "poor_tracking",
                    "processing_time_ms": 0., "is_cooldown": False}
        self.predictor.reset_state()
        self.last_prediction = timestamp
        result = self.service.predict(raw, mask, predictor=self.predictor, timestamps=times,
                                      width=width, height=height, end_time=end_time)
        result["segment"] = {"start": segment[0], "end": segment[1]}
        result.setdefault("status", "recognized" if result.get("sign") else "uncertain")
        if not result.get("sign"):
            result["top_k"] = []
        return result
