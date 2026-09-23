#!/usr/bin/env python3
"""
Valida landmarks extraídos — checa integridade dos .npz, distribuição, e outliers.

Uso:
    python scripts/validate_landmarks.py --dir data/landmarks
"""
import argparse
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_all(data_dir: Path) -> dict:
    """Valida todos os .npz no diretório."""
    npz_files = sorted(data_dir.rglob("*.npz"))
    logger.info(f"Encontrados {len(npz_files)} arquivos .npz")

    labels_by_path = {}
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            samples = json.load(f)
        labels_by_path = {sample["path"]: sample["label"] for sample in samples}

    stats = {"total": len(npz_files), "valid": 0, "invalid": 0, "empty": 0, "errors": []}
    frame_counts = []
    feature_dims = set()
    class_counts: dict[str, int] = {}

    for npz_path in npz_files:
        try:
            data = np.load(str(npz_path), allow_pickle=True)
            landmarks = data["landmarks"]
            relative_path = str(npz_path.relative_to(data_dir))
            label = labels_by_path.get(relative_path, npz_path.parent.name)

            if landmarks.ndim != 2:
                stats["errors"].append(f"{npz_path.name}: dimensão errada ({landmarks.shape})")
                stats["invalid"] += 1
                continue

            T, D = landmarks.shape
            if T == 0:
                stats["empty"] += 1
                continue

            feature_dims.add(D)
            frame_counts.append(T)
            class_counts[label] = class_counts.get(label, 0) + 1

            # Checar NaN/Inf
            if np.any(np.isnan(landmarks)) or np.any(np.isinf(landmarks)):
                stats["errors"].append(f"{npz_path.name}: contém NaN ou Inf")
                stats["invalid"] += 1
                continue

            # Checar valores extremos
            max_val = np.abs(landmarks).max()
            if max_val > 100:
                stats["errors"].append(f"{npz_path.name}: valor extremo ({max_val:.1f})")

            stats["valid"] += 1

        except Exception as e:
            stats["errors"].append(f"{npz_path.name}: {e}")
            stats["invalid"] += 1

    if frame_counts:
        stats["frame_stats"] = {
            "min": int(np.min(frame_counts)),
            "max": int(np.max(frame_counts)),
            "mean": float(np.mean(frame_counts)),
            "median": float(np.median(frame_counts)),
            "std": float(np.std(frame_counts)),
        }
    stats["feature_dims"] = sorted(feature_dims)
    stats["num_classes"] = len(class_counts)
    stats["samples_per_class"] = {
        "min": min(class_counts.values()) if class_counts else 0,
        "max": max(class_counts.values()) if class_counts else 0,
        "mean": np.mean(list(class_counts.values())) if class_counts else 0,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Validar landmarks extraídos")
    parser.add_argument("--dir", type=str, default="data/landmarks", help="Diretório de landmarks")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        logger.error(f"Diretório não encontrado: {data_dir}")
        return

    stats = validate_all(data_dir)

    logger.info("=" * 50)
    logger.info("RESULTADO DA VALIDAÇÃO")
    logger.info("=" * 50)
    logger.info(f"  Total: {stats['total']}")
    logger.info(f"  Válidos: {stats['valid']}")
    logger.info(f"  Inválidos: {stats['invalid']}")
    logger.info(f"  Vazios: {stats['empty']}")
    logger.info(f"  Classes: {stats['num_classes']}")
    logger.info(f"  Feature dims: {stats['feature_dims']}")

    if "frame_stats" in stats:
        fs = stats["frame_stats"]
        logger.info(f"  Frames: min={fs['min']}, max={fs['max']}, média={fs['mean']:.1f}, mediana={fs['median']:.0f}")

    if "samples_per_class" in stats:
        sc = stats["samples_per_class"]
        logger.info(f"  Amostras/classe: min={sc['min']}, max={sc['max']}, média={sc['mean']:.1f}")

    if stats["errors"]:
        logger.warning(f"  Erros ({len(stats['errors'])}):")
        for e in stats["errors"][:20]:
            logger.warning(f"    {e}")

    # Salvar relatório
    report_path = data_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Relatório salvo em {report_path}")


if __name__ == "__main__":
    main()
