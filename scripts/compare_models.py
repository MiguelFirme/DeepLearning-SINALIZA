#!/usr/bin/env python3
"""
Comparação lado a lado de múltiplos modelos treinados.

Uso:
    python scripts/compare_models.py --experiments experiments/bilstm_48f experiments/transformer_48f
"""
import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_experiment(exp_dir: Path) -> dict:
    """Carrega métricas de um experimento."""
    info = {"name": exp_dir.name}

    # Histórico
    history_path = exp_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        if history:
            best = min(history, key=lambda h: h.get("val_loss", float("inf")))
            info["best_epoch"] = best.get("epoch", "?")
            info["best_val_loss"] = best.get("val_loss", "?")
            info["best_val_top1"] = best.get("val_top1", "?")
            info["best_val_f1"] = best.get("val_f1_macro", "?")
            info["total_epochs"] = len(history)

    # Checkpoint
    ckpt_path = exp_dir / "best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        info["model_class"] = ckpt.get("model_class", "?")
        info["num_classes"] = ckpt.get("num_classes", "?")
        if "val_metrics" in ckpt:
            info.update({f"test_{k}": v for k, v in ckpt["val_metrics"].items()})

    return info


def main():
    parser = argparse.ArgumentParser(description="Comparar modelos")
    parser.add_argument("--experiments", nargs="+", required=True, help="Diretórios de experimentos")
    parser.add_argument("--output", type=str, default="reports/comparison.md")
    args = parser.parse_args()

    results = []
    for exp_path in args.experiments:
        exp_dir = Path(exp_path)
        if exp_dir.exists():
            results.append(load_experiment(exp_dir))
        else:
            logger.warning(f"Não encontrado: {exp_dir}")

    if not results:
        logger.error("Nenhum experimento encontrado.")
        return

    # Gerar tabela Markdown
    lines = [
        "# Comparação de Modelos — Sinaliza",
        "",
        "| Métrica | " + " | ".join(r["name"] for r in results) + " |",
        "|---------|" + "|".join(["------" for _ in results]) + "|",
    ]

    metrics = ["model_class", "total_epochs", "best_epoch", "best_val_loss", "best_val_top1", "best_val_f1"]
    labels = ["Modelo", "Épocas totais", "Melhor época", "Val Loss", "Val Top-1", "Val F1 (macro)"]

    for metric, label in zip(metrics, labels):
        vals = []
        for r in results:
            v = r.get(metric, "—")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    report = "\n".join(lines)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Comparação salva em {output_path}")
    print(report)


if __name__ == "__main__":
    main()
