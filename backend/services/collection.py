"""Coleta consentida dos mesmos landmarks que entram na inferência."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from uuid import uuid4

import numpy as np


class LandmarkCollector:
    def __init__(self, root, labels, app_version):
        self.root = Path(root)
        self.labels = set(labels) | {"_nenhum_sinal"}
        self.app_version = app_version
        self.cancel()

    def cancel(self):
        self.active = False
        self.frames = []
        self.metadata = {}
        return {"status": "cancelado"}

    def start(self, person, session, word, duration=4., consent=False, now=None):
        if consent is not True:
            raise ValueError("Confirme o consentimento antes de gravar")
        if self.active:
            raise ValueError("Já existe uma gravação em andamento")
        for value in (person, session):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
                raise ValueError("Pessoa e sessão devem usar IDs de até 64 letras, números, _ ou -")
        if word not in self.labels or not isinstance(word, str) or "/" in word or "\\" in word or word in (".", ".."):
            raise ValueError("Palavra fora do vocabulário de coleta")
        duration = float(duration)
        if not np.isfinite(duration) or not 1 <= duration <= 30:
            raise ValueError("Duração deve ficar entre 1 e 30 segundos")
        self.started = time.monotonic() if now is None else float(now)
        self.duration = duration
        self.frames = []
        self.metadata = {"person": person, "session": session, "word": word,
                         "date": datetime.now(timezone.utc).isoformat(), "app_version": self.app_version,
                         "consent": True, "origin": "novas", "format_version": 1}
        self.active = True
        return {"status": "gravando", "duration": duration}

    def feed(self, raw, mask, timestamp, width, height):
        if not self.active:
            return None
        if self.frames and self.frames[-1][3:] != (width, height):
            self.cancel()
            return {"status": "erro", "message": "Resolução mudou; a gravação foi descartada"}
        self.frames.append((float(timestamp), np.asarray(raw, dtype=np.float32).copy(),
                            np.asarray(mask, dtype=bool).copy(), int(width), int(height)))
        if timestamp - self.started >= self.duration:
            return self.finish()
        return None

    def finish(self):
        if not self.active:
            raise ValueError("Nenhuma gravação em andamento")
        self.active = False
        if len(self.frames) < 2:
            self.frames = []
            raise ValueError("Gravação sem quadros suficientes; tente novamente")
        times = np.asarray([frame[0] for frame in self.frames], dtype=np.float64)
        times -= times[0]
        if np.any(np.diff(times) <= 0):
            raise ValueError("Timestamps de coleta não são crescentes")
        raw = np.stack([frame[1] for frame in self.frames])
        mask = np.stack([frame[2] for frame in self.frames])
        width, height = self.frames[0][3:]
        meta = {**self.metadata, "width": width, "height": height,
                "fps": float(1 / np.median(np.diff(times))), "total_frames": len(times),
                "recorded_seconds": float(times[-1]), "requested_seconds": self.duration}
        directory = self.root / meta["person"] / meta["word"]
        directory.mkdir(parents=True, exist_ok=True)
        filename = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8] + ".npz"
        path = directory / filename
        with path.open("xb") as stream:
            np.savez_compressed(stream, landmarks=raw, mask=mask, timestamps=times,
                                metadata=json.dumps(meta, ensure_ascii=False))
        self.frames = []
        return {"status": "salvo", "path": str(path), "frames": len(times), "seconds": float(times[-1])}
