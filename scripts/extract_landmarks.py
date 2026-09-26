#!/usr/bin/env python3
"""
Extração em lote de landmarks via MediaPipe → cache .npz.

Cada .npz contém: landmarks (T, 346), mask (T, 4), metadata dict.
Nunca re-roda MediaPipe por época — o cache é permanente.

Uso:
    python scripts/extract_landmarks.py --input data/raw --output data/landmarks --workers 4
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Garante que os processos criados pelo multiprocessing no Windows também
# encontrem o pacote local ``ml``, independentemente do diretório de execução.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT_DIR = Path(os.environ.get("SINALIZA_VIDEO_DIR", PROJECT_ROOT / "data" / "raw"))
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "landmarks"


def extract_single(args_tuple: tuple) -> dict:
    """Extrai landmarks de um vídeo (executado em worker)."""
    video_path_str, output_path_str, config_dict = args_tuple
    video_path = Path(video_path_str)
    output_path = Path(output_path_str)

    if output_path.exists():
        return {"path": str(video_path), "status": "skipped", "frames": 0}

    try:
        from ml.features.landmarks import LandmarkExtractor, LandmarkConfig
        config = LandmarkConfig(**config_dict) if config_dict else LandmarkConfig()

        with LandmarkExtractor(config) as extractor:
            landmarks, masks, metadata = extractor.extract_video(video_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(output_path),
            landmarks=landmarks,
            mask=masks,
            metadata=np.array([json.dumps(metadata)]),
        )
        return {"path": str(video_path), "status": "ok", "frames": landmarks.shape[0]}

    except Exception as e:
        return {"path": str(video_path), "status": "error", "error": str(e)}


def collect_tasks(input_dir: Path, output_dir: Path) -> list[tuple]:
    """Coleta vídeos para processar."""
    tasks = []
    for video in sorted(input_dir.rglob("*")):
        if video.is_file() and video.suffix.lower() in VIDEO_EXTS:
            rel = video.relative_to(input_dir)
            npz_path = output_dir / rel.parent / (rel.stem + ".npz")
            tasks.append((str(video), str(npz_path), {}))
    return tasks


def build_manifest(output_dir: Path):
    """Constrói manifesto JSON das amostras extraídas."""
    samples = []
    for npz_path in sorted(output_dir.rglob("*.npz")):
        match = re.match(
            r"^(?P<label>.+)_Articulador(?P<numero>\d+)$",
            npz_path.stem,
            flags=re.IGNORECASE,
        )
        if match:
            label = match.group("label")
            signer = f"Articulador{match.group('numero')}"
        else:
            label = npz_path.parent.name
            signer = "unknown"
        samples.append({
            "path": str(npz_path.relative_to(output_dir)),
            "label": label,
            "signer": signer,
        })

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    logger.info(f"Manifesto salvo: {manifest_path} ({len(samples)} amostras)")


def main():
    parser = argparse.ArgumentParser(description="Extrair landmarks de vídeos")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Diretório de vídeos",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Diretório de saída .npz",
    )
    parser.add_argument("--workers", type=int, default=1, help="Workers paralelos")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(
            f"Diretório de vídeos não encontrado: {input_dir}. "
            "Informe --input ou defina SINALIZA_VIDEO_DIR."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = collect_tasks(input_dir, output_dir)
    logger.info(f"Total de vídeos: {len(tasks)}")

    # Processar
    results = []
    if args.workers <= 1:
        for i, task in enumerate(tasks):
            r = extract_single(task)
            results.append(r)
            if (i + 1) % 50 == 0:
                logger.info(f"Progresso: {i+1}/{len(tasks)}")
    else:
        with mp.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(extract_single, tasks)):
                results.append(r)
                if (i + 1) % 50 == 0:
                    logger.info(f"Progresso: {i+1}/{len(tasks)}")

    # Resumo
    ok = sum(1 for r in results if r["status"] == "ok")
    skip = sum(1 for r in results if r["status"] == "skipped")
    err = sum(1 for r in results if r["status"] == "error")
    logger.info(f"Resultado: {ok} OK, {skip} pulados, {err} erros")

    if err > 0:
        for r in results:
            if r["status"] == "error":
                logger.warning(f"  ERRO: {r['path']} — {r.get('error', '?')}")

    # Construir manifesto
    build_manifest(output_dir)


if __name__ == "__main__":
    main()
