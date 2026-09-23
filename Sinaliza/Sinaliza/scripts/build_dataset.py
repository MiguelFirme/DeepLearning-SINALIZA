#!/usr/bin/env python3
"""
Constrói dataset final: gera splits train/val/test e salva metadados.

Uso:
    python scripts/build_dataset.py --landmarks data/landmarks --output data/processed
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Construir dataset com splits")
    parser.add_argument("--landmarks", type=str, default="data/landmarks")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--strategy", type=str, default="stratified", choices=["stratified", "group"])
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from ml.data.split import DataSplitter, SplitConfig
    landmarks_dir = Path(args.landmarks)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carregar amostras do manifesto
    manifest_path = landmarks_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"Manifesto não encontrado: {manifest_path}. Execute extract_landmarks.py primeiro.")
        return

    with open(manifest_path, encoding="utf-8") as f:
        samples = json.load(f)

    logger.info(f"Total de amostras: {len(samples)}")

    # Construir label map
    labels = sorted(set(s["label"] for s in samples))
    label_map = {label: idx for idx, label in enumerate(labels)}
    logger.info(f"Total de classes: {len(label_map)}")

    # Split
    split_config = SplitConfig(
        strategy=args.strategy,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )
    splitter = DataSplitter(split_config)
    splits = splitter.split(samples)

    # Salvar
    splits_path = output_dir / "splits.json"
    splitter.save_split(splits, splits_path)

    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    logger.info(f"Label map salvo em {label_map_path}")

    # Copiar manifesto
    shutil.copy2(manifest_path, output_dir / "manifest.json")

    # Metadados
    meta = {
        "landmarks_dir": str(landmarks_dir),
        "num_samples": len(samples),
        "num_classes": len(label_map),
        "split_strategy": args.strategy,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "seed": args.seed,
    }
    with open(output_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Dataset construído com sucesso!")
    for name, idx in splits.items():
        logger.info(f"  {name}: {len(idx)} amostras")


if __name__ == "__main__":
    main()
