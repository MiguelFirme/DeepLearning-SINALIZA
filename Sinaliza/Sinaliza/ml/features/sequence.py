"""Processamento de sequências temporais: amostragem, padding, interpolação."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class SequenceConfig:
    target_frames: int = 48
    sampling_strategy: str = "uniform"  # uniform, interpolate, pad
    pad_value: float = 0.0

class SequenceProcessor:
    def __init__(self, config: SequenceConfig | None = None):
        self.config = config or SequenceConfig()

    def process(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (seq_fixo, mask) com shape (target, D) e (target,)."""
        T, D = seq.shape
        target = self.config.target_frames
        if T == 0:
            return np.zeros((target, D), dtype=np.float32), np.zeros(target, dtype=bool)
        if self.config.sampling_strategy == "interpolate" and T > 1:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, T)
            x_new = np.linspace(0, 1, target)
            try:
                f = interp1d(x_old, seq, axis=0, kind="linear", fill_value="extrapolate")
                return f(x_new).astype(np.float32), np.ones(target, dtype=bool)
            except Exception:
                pass
        # Uniform sampling (default)
        if T >= target:
            idx = np.linspace(0, T-1, target, dtype=int)
            return seq[idx].astype(np.float32), np.ones(target, dtype=bool)
        else:
            idx = np.linspace(0, T-1, target, dtype=int)
            return seq[idx].astype(np.float32), np.ones(target, dtype=bool)
