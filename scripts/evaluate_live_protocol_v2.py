#!/usr/bin/env python3
"""Avalia um checkpoint v2 com o protocolo da webcam, não com a janela ideal.

A validação do treino classifica uma única janela centrada no gesto, escolhida
olhando o vídeo inteiro. Na webcam o backend recebe quadros a ~6–12 fps e
decide sozinho quando e o que mostrar. Este script passa cada vídeo de um split
pela mesma sessão do backend (`TemporalWindowSessionV2`), na cadência pedida, e
conta o que o usuário veria:

- primeira_correta: a primeira palavra mostrada é a correta;
- alguma_correta: a palavra correta aparece em algum momento;
- palavras_erradas: palavras aceitas que não são o rótulo (falsos positivos);
- sem_resposta: nenhum sinal aceito no vídeo inteiro.

Uso (Git Bash, na raiz):
    ./.venv/Scripts/python.exe scripts/evaluate_live_protocol_v2.py \\
        --checkpoint experiments/transformer_v2_mixed23_seed42/best.pt \\
        --data-dir data/landmarks/mixed_23 --processed-dir data/processed/mixed_23 \\
        --split val --fps 9 15
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import settings
from backend.services.prediction import PredictionService
from scripts.simulate_live_fps import npz_frames, simulate_frames


def evaluate(service, data_dir: Path, rows, fps: float):
    per_video, confusions = [], Counter()
    for row in rows:
        result = simulate_frames(service, npz_frames(data_dir / row["path"], fps))
        accepted = [w["accepted_sign"] for w in result["windows"] if w.get("accepted_sign")]
        wrong = [sign for sign in accepted if sign != row["label"]]
        confusions.update((row["label"], sign) for sign in wrong)
        per_video.append({
            "path": row["path"], "label": row["label"], "origin": row.get("origin", "?"),
            "accepted": accepted,
            "first_correct": bool(accepted) and accepted[0] == row["label"],
            "any_correct": row["label"] in accepted,
            "wrong": len(wrong),
            "seconds": result["windows"][-1]["end_seconds"] if result["windows"] else 0.,
        })
    return per_video, confusions


def summarize(videos):
    n = len(videos)
    if n == 0:
        return {}
    minutes = sum(v["seconds"] for v in videos) / 60 or 1
    return {
        "videos": n,
        "primeira_correta": round(sum(v["first_correct"] for v in videos) / n, 3),
        "alguma_correta": round(sum(v["any_correct"] for v in videos) / n, 3),
        "sem_resposta": round(sum(not v["accepted"] for v in videos) / n, 3),
        "palavras_erradas": sum(v["wrong"] for v in videos),
        "palavras_erradas_por_minuto": round(sum(v["wrong"] for v in videos) / minutes, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--fps", type=float, nargs="+", default=[9.0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    settings.device = "cpu"
    service = PredictionService()
    service.load_model(args.checkpoint)
    if not service.is_loaded or service.pipeline_version != 2:
        raise SystemExit("Informe um checkpoint v2")
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((args.processed_dir / "splits.json").read_text(encoding="utf-8"))
    rows = [manifest[i] for i in splits[args.split]]
    mode = "segmentado" if getattr(service.pipeline_cfg, "segmentation", False) else "janela deslizante + EMA"
    report = {"checkpoint": args.checkpoint, "split": args.split, "modo": mode, "por_fps": {}}
    for fps in args.fps:
        videos, confusions = evaluate(service, args.data_dir, rows, fps)
        origins = sorted({v["origin"] for v in videos})
        summary = {"geral": summarize(videos),
                   **{origin: summarize([v for v in videos if v["origin"] == origin]) for origin in origins},
                   "confusoes_mais_frequentes": [f"{a} → {b}: {c}" for (a, b), c in confusions.most_common(10)]}
        report["por_fps"][str(fps)] = {"resumo": summary, "videos": videos}
        print(f"\n=== {args.split} | {fps:g} fps | {mode} ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
