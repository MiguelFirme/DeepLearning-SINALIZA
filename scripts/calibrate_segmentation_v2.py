#!/usr/bin/env python3
"""Confere os limiares de `SegmentationConfig` nos .npz reais antes do treino.

Bom sinal: quase todos os vídeos com um trecho detectado, duração típica entre
0,6 e 2,5 s, repouso (p95) bem abaixo de `off_speed` e pico do gesto (p50)
bem acima de `on_speed`. Rode também nos arquivos de `data/coleta/` (webcam).

    ./.venv/Scripts/python.exe scripts/calibrate_segmentation_v2.py --data-dir data/landmarks/mixed_23
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data.dataset_v2 import read_metadata
from ml.features.segmentation_v2 import SegmentationConfig, main_segment, wrist_speed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    cfg = SegmentationConfig()
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    stats = defaultdict(lambda: {"videos": 0, "sem_trecho": 0, "duracao": [], "repouso": [], "pico": []})
    for row in manifest:
        with np.load(args.data_dir / row["path"], allow_pickle=True) as data:
            meta = read_metadata(data)
            raw, mask = data["landmarks"], data["mask"].astype(bool)
            fps = float(meta.get("fps") or 30)
            times = data["timestamps"] if "timestamps" in data else np.arange(len(raw)) / fps
        width, height = float(meta.get("width") or 1), float(meta.get("height") or 1)
        origin = row.get("origin", "?")
        item = stats[origin]
        item["videos"] += 1
        speed = wrist_speed(raw, mask, times, width, height, cfg.smooth_seconds)
        rest = speed[(times - times[0] > .25) & (times - times[0] < .6)]
        if len(rest):
            item["repouso"].append(float(np.percentile(rest, 95)))
        item["pico"].append(float(speed.max()))
        segment = main_segment(raw, mask, times, width, height, cfg)
        if segment is None:
            item["sem_trecho"] += 1
        else:
            item["duracao"].append(segment[1] - segment[0])
    print(f"limiares: on={cfg.on_speed} off={cfg.off_speed} (larguras de ombro/s)")
    for origin, item in stats.items():
        q = lambda values, p: round(float(np.percentile(values, p)), 2) if values else None
        print(json.dumps({
            "origem": origin, "videos": item["videos"], "sem_trecho": item["sem_trecho"],
            "duracao_s_p10_p50_p90": [q(item["duracao"], p) for p in (10, 50, 90)],
            "repouso_inicio_p95_dos_videos": q(item["repouso"], 95),
            "pico_do_gesto_p10_p50": [q(item["pico"], p) for p in (10, 50)],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
