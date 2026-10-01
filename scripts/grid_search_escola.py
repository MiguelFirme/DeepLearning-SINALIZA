#!/usr/bin/env python3
"""Busca pequena de Bi-LSTM com validação por pessoa e logs persistentes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.vocabulary import read_annotations, select_essential_categories

logger = logging.getLogger(__name__)


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_person_splits(spec: dict, processed_root: Path) -> dict:
    """Treino A1; validação A2 real; teste A3 real; refit A1+A2."""
    data_dir = PROJECT_ROOT / spec["data_dir"]
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotations = read_annotations(PROJECT_ROOT / spec["annotations"])
    selected = select_essential_categories(set(annotations), spec["categories"])
    if len(selected) != spec["expected_classes"]:
        raise ValueError(f"Esperadas {spec['expected_classes']} classes, encontradas {len(selected)}")
    signers = (spec["train_signer"], spec["validation_signer"], spec["test_signer"])
    if len(set(signers)) != 3:
        raise ValueError("Treino, validação e teste precisam de pessoas distintas")

    by_path = {sample["path"]: sample for sample in manifest}
    if len(by_path) != len(manifest):
        raise ValueError("Manifesto contém caminhos duplicados")
    original_counts = {(label, signer): 0 for label in selected for signer in signers}
    per_class = {label: 0 for label in selected}
    for sample in manifest:
        label = sample["label"]
        if label not in selected:
            raise ValueError(f"Classe fora do recorte no manifesto: {label}")
        signer = sample.get("signer")
        if signer not in signers:
            raise ValueError(f"Articulador inesperado: {signer}")
        if not (data_dir / sample["path"]).is_file():
            raise FileNotFoundError(data_dir / sample["path"])
        per_class[label] += 1
        if sample.get("is_synthetic", False):
            source = by_path.get(sample.get("source_path"))
            if (source is None or source.get("is_synthetic", False)
                    or source["label"] != label or source.get("signer") != signer):
                raise ValueError(f"Origem sintética inválida: {sample['path']}")
        else:
            original_counts[label, signer] += 1
    if set(per_class.values()) != {10} or set(original_counts.values()) != {1}:
        raise ValueError("Esperados dez arquivos e um original por articulador em cada classe")

    train_signer, val_signer, test_signer = signers
    tune = {"train": [], "val": [], "test": []}
    refit = {"train": [], "val": [], "test": []}
    for index, sample in enumerate(manifest):
        signer = sample["signer"]
        real = not sample.get("is_synthetic", False)
        if signer == train_signer:
            tune["train"].append(index)
            refit["train"].append(index)
        elif signer == val_signer:
            if real:
                tune["val"].append(index)
            refit["train"].append(index)
        elif real:
            tune["test"].append(index)
            refit["test"].append(index)
    if len(tune["val"]) != len(selected) or len(tune["test"]) != len(selected):
        raise ValueError("Validação e teste devem ter um vídeo real por classe")
    if set(tune["train"]) & set(tune["val"] + tune["test"]):
        raise ValueError("Vazamento entre treino e avaliação")

    label_map = {label: index for index, label in enumerate(sorted(selected))}
    for name, split in (("tune", tune), ("refit", refit)):
        target = processed_root / name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, target / "manifest.json")
        save_json(target / "splits.json", split)
        save_json(target / "label_map.json", label_map)
        save_json(target / "dataset_meta.json", {
            "strategy": "person_grid", "split_sizes": {k: len(v) for k, v in split.items()},
            "train_signers": [train_signer] if name == "tune" else [train_signer, val_signer],
            "validation_signer": val_signer if name == "tune" else None,
            "test_signer": test_signer,
            "test_real_only": True, "num_classes": len(label_map),
        })
    return {"manifest_sha256": file_hash(manifest_path),
            "tune_sizes": {k: len(v) for k, v in tune.items()},
            "refit_sizes": {k: len(v) for k, v in refit.items()},
            "classes": sorted(selected)}


def trial_config(base_config: Path, hidden: int, layers: int, dropout: float,
                 lr: float, epochs: int) -> dict:
    return {
        "_base_": str(base_config.resolve()),
        "model": {"hidden_dim": hidden, "num_layers": layers, "dropout": dropout},
        "training": {"epochs": epochs, "learning_rate": lr,
                     "monitor": "val_loss", "early_stopping": False},
    }


def write_config(path: Path, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"Configuração já existente diverge: {path}")
    path.write_text(content, encoding="utf-8")


def run_training(spec: dict, config: Path, processed: Path, save_dir: Path,
                 log_root: Path, name: str, epochs: int, final: bool) -> dict:
    artifact_dir = save_dir / name
    record_path = log_root / save_dir.name / name / "run_record.json"
    required = [artifact_dir / "best.pt", artifact_dir / "history.json", record_path]
    if final:
        required.append(artifact_dir / "test_metrics.json")
    if all(path.exists() for path in required):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "completed":
            logger.info("Reutilizando %s", name)
            return {"returncode": 0, "reused": True}

    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"),
           "--config", str(config), "--data-dir", str(PROJECT_ROOT / spec["data_dir"]),
           "--processed-dir", str(processed), "--save-dir", str(save_dir),
           "--log-dir", str(log_root), "--experiment", name,
           "--epochs", str(epochs), "--seed", str(spec["seed"]), "--num-workers", "0"]
    cmd += ["--test-checkpoint", "last"] if final else ["--skip-test"]
    output_path = log_root / "subprocess" / save_dir.name / f"{name}.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Treinando %s; saída em %s", name, output_path)
    with output_path.open("w", encoding="utf-8") as output:
        process = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=output,
                                 stderr=subprocess.STDOUT, check=False)
    if not all(path.exists() for path in required):
        raise RuntimeError(f"Treino {name} incompleto (exit={process.returncode}); veja {output_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise RuntimeError(f"Treino {name} falhou (exit={process.returncode}); veja {output_path}")
    if process.returncode:
        logger.warning("%s retornou %s após salvar artefatos completos", name, process.returncode)
    return {"returncode": process.returncode, "reused": False}


def trial_result(log_root: Path, save_dir: Path, name: str, params: dict,
                 stage: str, process: dict) -> dict:
    record = json.loads((log_root / save_dir.name / name / "run_record.json")
                        .read_text(encoding="utf-8"))
    history = json.loads((save_dir / name / "history.json").read_text(encoding="utf-8"))
    best_epoch = int(record["result"]["best_epoch"])
    best = history[best_epoch - 1]
    return {"name": name, "stage": stage, **params,
            "best_epoch": best_epoch, "total_epochs": len(history),
            "val_top1": best["val_top1"], "val_top3": best["val_top3"],
            "val_f1_macro": best["val_f1_macro"], "val_loss": best["val_loss"],
            "train_top1": best["train_top1"],
            "parameter_count": record["resolved"]["parameter_count"],
            "elapsed_seconds": record["result"]["elapsed_seconds"], **process}


def ranking(row: dict) -> tuple:
    return (-row["val_top1"], -row["val_f1_macro"], row["val_loss"],
            row["parameter_count"], row["name"])


def save_results(log_root: Path, rows: list[dict]):
    save_json(log_root / "grid_results.json", rows)
    if rows:
        with (log_root / "grid_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/grid_escola_23.json")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    spec_path = PROJECT_ROOT / args.spec
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    run_name = spec["run_name"]
    experiment_root = PROJECT_ROOT / "experiments" / run_name
    processed_root = PROJECT_ROOT / "data" / "processed" / run_name
    log_root = PROJECT_ROOT / "logs" / run_name
    experiment_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_root / "grid.log", encoding="utf-8")])

    dataset = prepare_person_splits(spec, processed_root)
    identity = {"spec_sha256": file_hash(spec_path), **dataset}
    identity_path = log_root / "identity.json"
    if identity_path.exists():
        old = json.loads(identity_path.read_text(encoding="utf-8"))
        if old != identity:
            raise ValueError("Spec ou dataset mudou; use um novo run_name")
    save_json(identity_path, identity)
    logger.info("Splits: tune=%s, refit=%s", dataset["tune_sizes"], dataset["refit_sizes"])
    if args.prepare_only:
        return

    state_path = log_root / "grid_state.json"
    state = {"status": "running", "started_at": datetime.now().isoformat(timespec="seconds"),
             "run_name": run_name, "trials": {}}
    save_json(state_path, state)
    results: list[dict] = []
    base = PROJECT_ROOT / spec["base_config"]
    trials_dir = experiment_root / "trials"
    try:
        arch_params = list(itertools.product(spec["hidden_sizes"], spec["num_layers"],
                                             spec["learning_rates"]))
        for number, (hidden, layers, lr) in enumerate(arch_params, start=1):
            name = f"arch_{number:02d}"
            params = {"hidden_size": hidden, "num_layers": layers,
                      "learning_rate": lr, "dropout": spec["initial_dropout"]}
            config = experiment_root / "configs" / f"{name}.yaml"
            write_config(config, trial_config(base, hidden, layers, params["dropout"],
                                              lr, spec["epochs"]))
            state["trials"][name] = "running"
            save_json(state_path, state)
            process = run_training(spec, config, processed_root / "tune", trials_dir,
                                   log_root, name, spec["epochs"], final=False)
            row = trial_result(log_root, trials_dir, name, params, "architecture", process)
            results.append(row)
            save_results(log_root, results)
            state["trials"][name] = "completed"
            save_json(state_path, state)
            logger.info("%s: val_top1=%.4f val_loss=%.4f", name, row["val_top1"], row["val_loss"])

        best_arch = min(results, key=ranking)
        dropout_rows = [best_arch]
        for number, dropout in enumerate(spec["dropout_candidates"], start=1):
            if dropout == spec["initial_dropout"]:
                continue
            name = f"drop_{number:02d}"
            params = {"hidden_size": best_arch["hidden_size"],
                      "num_layers": best_arch["num_layers"],
                      "learning_rate": best_arch["learning_rate"], "dropout": dropout}
            config = experiment_root / "configs" / f"{name}.yaml"
            write_config(config, trial_config(base, params["hidden_size"],
                                              params["num_layers"], dropout,
                                              params["learning_rate"], spec["epochs"]))
            state["trials"][name] = "running"
            save_json(state_path, state)
            process = run_training(spec, config, processed_root / "tune", trials_dir,
                                   log_root, name, spec["epochs"], final=False)
            row = trial_result(log_root, trials_dir, name, params, "dropout", process)
            results.append(row)
            dropout_rows.append(row)
            save_results(log_root, results)
            state["trials"][name] = "completed"
            save_json(state_path, state)
            logger.info("%s: val_top1=%.4f val_loss=%.4f", name, row["val_top1"], row["val_loss"])

        winner = min(dropout_rows, key=ranking)
        selection = {"criterion": "val_top1 desc, val_f1_macro desc, val_loss asc",
                     "selected_trial": winner, "test_not_used_for_selection": True}
        save_json(log_root / "selection.json", selection)
        logger.info("Selecionado %s: %s", winner["name"], winner)

        final_config = experiment_root / "configs" / "final_refit.yaml"
        write_config(final_config, trial_config(base, winner["hidden_size"],
                                                winner["num_layers"], winner["dropout"],
                                                winner["learning_rate"], winner["best_epoch"]))
        final_dir = experiment_root / "final"
        state["status"] = "final_training"
        save_json(state_path, state)
        process = run_training(spec, final_config, processed_root / "refit", final_dir,
                               log_root, "selected_refit", winner["best_epoch"], final=True)
        metrics = json.loads((final_dir / "selected_refit" / "test_metrics.json")
                             .read_text(encoding="utf-8"))
        final = {"selected_trial": winner["name"], "hyperparameters": {
            key: winner[key] for key in ("hidden_size", "num_layers", "learning_rate", "dropout")},
            "refit_epochs": winner["best_epoch"], "test_signer": spec["test_signer"],
            "test_metrics": metrics, "process": process,
            "completed_at": datetime.now().isoformat(timespec="seconds")}
        save_json(log_root / "final_result.json", final)
        state["status"] = "completed"
        state["completed_at"] = final["completed_at"]
        save_json(state_path, state)
        logger.info("Teste final: %s", json.dumps(metrics, ensure_ascii=False))
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        save_json(state_path, state)
        logger.exception("Busca interrompida")
        raise


if __name__ == "__main__":
    main()
