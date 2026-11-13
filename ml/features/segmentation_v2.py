"""Segmentação do sinal pelo movimento dos punhos, igual no treino e ao vivo.

A validação v2 sempre classificava uma janela centrada no gesto, escolhida com o
vídeo inteiro disponível. Ao vivo a janela terminava "agora", a cada 0,4 s, e
quase sempre pegava o sinal pela metade ou só repouso. Este módulo define um
único critério de início/fim em segundos, usado pelo dataset e pela sessão da
webcam, para que a janela classificada ao vivo seja a mesma da validação.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Posições na lista de 25 pontos de pose do vetor bruto de 346 valores.
SHOULDERS = (5, 6)
WRISTS = (9, 10)


@dataclass
class SegmentationConfig:
    on_speed: float = 0.9      # larguras de ombro por segundo para iniciar
    off_speed: float = 0.45    # abaixo disso o punho é considerado parado
    quiet_seconds: float = 0.4  # repouso necessário para encerrar o trecho
    min_seconds: float = 0.3   # trechos menores são ruído
    max_seconds: float = 3.0   # trechos maiores não cabem na janela de 3,2 s
    smooth_seconds: float = 0.2


def wrist_speed(raw, mask, times, width, height, smooth_seconds=0.2):
    """Velocidade máxima dos punhos relativa aos ombros (larguras de ombro/s)."""
    raw = np.asarray(raw, dtype=np.float32)
    times = np.asarray(times, dtype=np.float64)
    n = len(raw)
    speed = np.zeros(n)
    if n < 2:
        return speed
    pose = raw[:, 126:226].reshape(n, 25, 4)[:, :, :2] * np.array([width / height, 1.0])
    shoulder_width = np.linalg.norm(pose[:, SHOULDERS[0]] - pose[:, SHOULDERS[1]], axis=1)
    ok = np.asarray(mask, dtype=bool)[:, 2] & np.isfinite(shoulder_width) & (shoulder_width > 1e-3)
    if ok.sum() < 2:
        return speed
    rel = (pose[:, list(WRISTS)] - pose[:, list(SHOULDERS)]) / np.median(shoulder_width[ok])
    # Deslocamento ao longo de ~smooth_seconds, não soma de passos quadro a
    # quadro: o tremor do MediaPipe não cresce com o FPS e o valor é o mesmo
    # a 9 ou a 30 fps.
    span = max(smooth_seconds, 1e-3)
    # Último quadro com pelo menos `span` segundos de distância: o intervalo
    # nunca é menor que span, nem com FPS baixo nem no começo do buffer.
    previous = np.searchsorted(times, times - span, side="right") - 1
    has_history = previous >= 0
    previous = previous.clip(0)
    elapsed = np.maximum(times - times[previous], 1e-3)
    displacement = np.linalg.norm(rel - rel[previous], axis=-1).max(axis=1)
    speed = displacement / elapsed
    speed[~(ok & ok[previous] & has_history)] = 0
    return speed


def motion_segments(raw, mask, times, width, height, cfg: SegmentationConfig | None = None,
                    require_closed: bool = True):
    """Lista de (início, fim) em segundos, com histerese e repouso mínimo.

    Com ``require_closed`` (uso ao vivo), só devolve trechos cujo repouso final
    já foi observado; no dataset o fim do vídeo também fecha o trecho.
    """
    cfg = cfg or SegmentationConfig()
    times = np.asarray(times, dtype=np.float64)
    speed = wrist_speed(raw, mask, times, width, height, cfg.smooth_seconds)
    segments, start, last_active = [], None, None
    for t, v in zip(times, speed):
        if start is None:
            if v >= cfg.on_speed:
                start, last_active = t, t
            continue
        if v >= cfg.off_speed:
            last_active = t
        elif t - last_active >= cfg.quiet_seconds:
            segments.append((start, last_active))
            start = None
    if start is not None and not require_closed:
        segments.append((start, float(times[-1])))
    return [(float(a), float(b)) for a, b in segments
            if cfg.min_seconds <= b - a <= cfg.max_seconds]


def main_segment(raw, mask, times, width, height, cfg: SegmentationConfig | None = None):
    """Trecho principal de um vídeo de sinal isolado: o de maior duração."""
    segments = motion_segments(raw, mask, times, width, height, cfg, require_closed=False)
    if not segments:
        return None
    return max(segments, key=lambda seg: seg[1] - seg[0])


def centered_window_end(segment, window_seconds):
    """Mesmo cálculo de ``window_end`` da validação: janela centrada no trecho."""
    start, end = segment
    return (start + end + window_seconds) / 2
