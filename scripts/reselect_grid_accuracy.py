#!/usr/bin/env python3
"""Reavalia a grade pela acurácia de validação, preservando a primeira análise."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.grid_search_escola import (
    prepare_person_splits, ranking, run_training, save_json, save_results,
    trial_config, write_config,
)

logger = logging.getLogger(__name__)


def peak_validation_row(history: list[dict], original: dict) -> dict:
    """Escolhe época só por validação: top-1, F1, depois perda."""
    best = min(history, key=lambda item: (
        -item["val_top1"], -item["val_f1_macro"], item["val_loss"], item["epoch"],
    ))
    return {
        **original,
        "best_epoch": best["epoch"],
        "val_top1": best["val_top1"],
        "val_top3": best["val_top3"],
        "val_f1_macro": best["val_f1_macro"],
        "val_loss": best["val_loss"],
        "train_top1": best["train_top1"],
        "checkpoint_epoch_by_val_loss": original.get("best_epoch"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/grid_escola_23.json")
    args = parser.parse_args()
    spec = json.loads((PROJECT_ROOT / args.spec).read_text(encoding="utf-8"))
    root = PROJECT_ROOT / "experiments" / spec["run_name"]
    log_root = PROJECT_ROOT / "logs" / spec["run_name"]
    review = log_root / "accuracy_review"
    review.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(review / "review.log", encoding="utf-8")])
    processed = PROJECT_ROOT / "data" / "processed" / spec["run_name"]
    prepare_person_splits(spec, processed)

    initial = json.loads((log_root / "grid_results.json").read_text(encoding="utf-8"))
    architecture_rows = [row for row in initial if row["stage"] == "architecture"]
    if len(architecture_rows) != 8:
        raise ValueError("A grade original precisa ter oito arquiteturas completas")
    rows = []
    for row in architecture_rows:
        history = json.loads((root / "trials" / row["name"] / "history.json")
                             .read_text(encoding="utf-8"))
        rows.append(peak_validation_row(history, row))
    best_arch = min(rows, key=ranking)
    logger.info("Arquitetura por pico de validação: %s", best_arch)
    save_results(review, rows)

    dropout_rows = [best_arch]
    trials_dir = root / "accuracy_trials"
    base = PROJECT_ROOT / spec["base_config"]
    for number, dropout in enumerate(spec["dropout_candidates"], start=1):
        if dropout == spec["initial_dropout"]:
            continue
        name = f"accuracy_drop_{number:02d}"
        config = root / "configs" / f"{name}.yaml"
        write_config(config, trial_config(base, best_arch["hidden_size"],
                                          best_arch["num_layers"], dropout,
                                          best_arch["learning_rate"], spec["epochs"]))
        process = run_training(spec, config, processed / "tune", trials_dir,
                               log_root, name, spec["epochs"], final=False)
        record = json.loads((log_root / trials_dir.name / name / "run_record.json")
                            .read_text(encoding="utf-8"))
        history = json.loads((trials_dir / name / "history.json").read_text(encoding="utf-8"))
        row = peak_validation_row(history, {
            "name": name, "stage": "dropout", "hidden_size": best_arch["hidden_size"],
            "num_layers": best_arch["num_layers"],
            "learning_rate": best_arch["learning_rate"], "dropout": dropout,
            "best_epoch": record["result"]["best_epoch"],
            "total_epochs": len(history),
            "parameter_count": record["resolved"]["parameter_count"],
            "elapsed_seconds": record["result"]["elapsed_seconds"], **process,
        })
        rows.append(row)
        dropout_rows.append(row)
        save_results(review, rows)
        logger.info("%s: peak val_top1=%.4f época=%s", name, row["val_top1"], row["best_epoch"])

    winner = min(dropout_rows, key=ranking)
    selection = {
        "criterion": "max val_top1, max val_f1_macro, min val_loss",
        "selected_trial": winner, "refit_epochs": spec["epochs"],
        "test_signer_previously_seen_in_other_experiments": True,
        "selected_without_new_test_metrics": True,
    }
    save_json(review / "selection.json", selection)
    logger.info("Selecionado: %s", winner)

    final_config = root / "configs" / "accuracy_refit.yaml"
    write_config(final_config, trial_config(base, winner["hidden_size"],
                                            winner["num_layers"], winner["dropout"],
                                            winner["learning_rate"], spec["epochs"]))
    final_dir = root / "accuracy_final"
    process = run_training(spec, final_config, processed / "refit", final_dir,
                           log_root, "selected_refit", spec["epochs"], final=True)
    metrics = json.loads((final_dir / "selected_refit" / "test_metrics.json")
                         .read_text(encoding="utf-8"))
    final = {
        "selected_trial": winner["name"], "hyperparameters": {
            key: winner[key] for key in ("hidden_size", "num_layers", "learning_rate", "dropout")},
        "validation": {key: winner[key] for key in ("val_top1", "val_f1_macro", "val_loss", "best_epoch")},
        "refit_epochs": spec["epochs"], "test_metrics": metrics,
        "process": process, "prior_test_seen": True,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(review / "final_result.json", final)
    logger.info("Teste exploratório após correção: %s", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
