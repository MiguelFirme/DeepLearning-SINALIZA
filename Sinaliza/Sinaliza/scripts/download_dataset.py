#!/usr/bin/env python3
"""
Download do dataset V-Librasil do Kaggle.

Requisitos:
    pip install kaggle
    Configurar ~/.kaggle/kaggle.json

Uso:
    python scripts/download_dataset.py --output data/raw
"""
import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_SLUG = "vlibrasil/v-librasil"  # Ajustar conforme o slug real no Kaggle


def download_kaggle(slug: str, output_dir: Path):
    """Faz download via Kaggle CLI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", slug, "-p", str(output_dir), "--unzip",
    ]
    logger.info(f"Executando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Erro no download:\n{result.stderr}")
        raise RuntimeError("Falha no download do Kaggle.")
    logger.info(f"Dataset baixado em {output_dir}")


def scan_dataset(data_dir: Path) -> dict:
    """Escaneia o dataset e retorna estatísticas."""
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    classes: dict[str, int] = {}
    total_videos = 0

    for item in sorted(data_dir.rglob("*")):
        if item.is_file() and item.suffix.lower() in video_exts:
            label = item.parent.name
            classes[label] = classes.get(label, 0) + 1
            total_videos += 1

    stats = {
        "total_videos": total_videos,
        "total_classes": len(classes),
        "avg_per_class": total_videos / max(len(classes), 1),
        "min_per_class": min(classes.values()) if classes else 0,
        "max_per_class": max(classes.values()) if classes else 0,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download V-Librasil")
    parser.add_argument("--output", type=str, default="data/raw", help="Diretório de saída")
    parser.add_argument("--slug", type=str, default=DATASET_SLUG, help="Slug Kaggle")
    parser.add_argument("--scan-only", action="store_true", help="Apenas escanear sem baixar")
    args = parser.parse_args()

    output = Path(args.output)

    if not args.scan_only:
        logger.info(f"Baixando dataset '{args.slug}' para {output}...")
        download_kaggle(args.slug, output)

    logger.info("Escaneando dataset...")
    stats = scan_dataset(output)
    logger.info(f"Estatísticas do dataset:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
