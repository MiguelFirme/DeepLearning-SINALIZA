#!/usr/bin/env python3
"""Executa validação cruzada por articulador sem compartilhar pesos.

Exemplo:
    python scripts/run_signer_cv.py --max-classes 50 --epochs 50
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SIGNERS = ("Articulador1", "Articulador2", "Articulador3")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def save_json_atomic(path: Path, payload: dict):
    """Grava estado de forma atômica para permitir retomada após interrupção."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def run_command(command: list[str], dry_run: bool = False):
    logger.info("> %s", subprocess.list2cmdline(command))
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def aggregate_results(results: dict[str, dict]) -> dict:
    metric_names = (
        "loss", "top1", "top3", "top5", "f1_macro", "f1_weighted",
        "precision_macro", "recall_macro",
    )
    aggregate: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        values = [float(metrics[metric]) for metrics in results.values() if metric in metrics]
        if values:
            aggregate[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return aggregate


def save_summary(run_dir: Path, run_config: dict, results: dict[str, dict]):
    summary = {
        "run_config": run_config,
        "completed_rounds": sorted(results),
        "rounds": results,
        "aggregate": aggregate_results(results),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json_atomic(run_dir / "summary.json", summary)

    if results:
        metric_names = sorted({key for metrics in results.values() for key in metrics})
        with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["test_signer", *metric_names])
            writer.writeheader()
            for signer, metrics in sorted(results.items()):
                writer.writerow({"test_signer": signer, **metrics})
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Rodar três holdouts por articulador")
    parser.add_argument("--config", default="configs/bilstm_focus.yaml")
    parser.add_argument("--data-dir", default="data/landmarks")
    parser.add_argument("--processed-root", default="data/processed/signer_cv")
    parser.add_argument("--run-root", default="experiments")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--extra-keep", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--round", choices=["all", "1", "2", "3"], default="all")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Gera e valida os splits, mas não inicia treinamento",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra os comandos sem escrever splits nem iniciar treinamento",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Treina novamente mesmo quando test_metrics.json já existe",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_classes is not None and args.max_classes <= 0:
        raise SystemExit("--max-classes deve ser maior que zero")
    if args.epochs <= 0:
        raise SystemExit("--epochs deve ser maior que zero")

    class_tag = str(args.max_classes) if args.max_classes is not None else "all"
    config_tag = Path(args.config).stem
    run_name = args.run_name or (
        f"{config_tag}_signer_cv_{class_tag}classes_{args.epochs}epochs_seed{args.seed}"
    )
    processed_root = Path(args.processed_root).expanduser().resolve() / run_name
    run_dir = Path(args.run_root).expanduser().resolve() / run_name
    selected_rounds = (1, 2, 3) if args.round == "all" else (int(args.round),)

    run_config = {
        "config": str(Path(args.config)),
        "data_dir": str(Path(args.data_dir)),
        "processed_root": str(processed_root),
        "run_dir": str(run_dir),
        "max_classes": args.max_classes,
        "annotations": args.annotations,
        "extra_keep": args.extra_keep,
        "epochs": args.epochs,
        "seed": args.seed,
        "selected_rounds": list(selected_rounds),
        "signers": list(SIGNERS),
    }
    state_path = run_dir / "run_state.json"
    state = {
        "status": "dry_run" if args.dry_run else "running",
        "config": run_config,
        "rounds": {},
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if not args.dry_run:
        save_json_atomic(state_path, state)

    results: dict[str, dict] = {}
    # Resultados anteriores também entram no resumo ao retomar.
    for signer in SIGNERS:
        metrics_path = run_dir / signer.lower() / "test_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as handle:
                results[signer] = json.load(handle)

    try:
        for round_number in selected_rounds:
            test_signer = SIGNERS[round_number - 1]
            train_signers = [signer for signer in SIGNERS if signer != test_signer]
            round_name = test_signer.lower()
            processed_dir = processed_root / round_name
            experiment_dir = run_dir / round_name
            metrics_path = experiment_dir / "test_metrics.json"

            state["rounds"][test_signer] = {
                "status": "preparing",
                "train_signers": train_signers,
                "test_signer": test_signer,
                "processed_dir": str(processed_dir),
                "experiment_dir": str(experiment_dir),
            }
            if not args.dry_run:
                save_json_atomic(state_path, state)

            build_command = [
                sys.executable,
                str(SCRIPTS_DIR / "build_dataset.py"),
                "--landmarks", str(Path(args.data_dir)),
                "--output", str(processed_dir),
                "--strategy", "signer_holdout",
                "--test-signer", test_signer,
                "--train-signers", *train_signers,
                "--seed", str(args.seed),
                "--annotations", str(Path(args.annotations)),
            ]
            if args.extra_keep:
                build_command.extend(["--extra-keep", *args.extra_keep])
            if args.max_classes is not None:
                build_command.extend(["--max-classes", str(args.max_classes)])
            run_command(build_command, args.dry_run)

            if args.prepare_only:
                state["rounds"][test_signer]["status"] = "prepared"
                if not args.dry_run:
                    save_json_atomic(state_path, state)
                continue

            if metrics_path.exists() and not args.force:
                logger.info("%s já concluído; reutilizando métricas.", test_signer)
                with open(metrics_path, encoding="utf-8") as handle:
                    results[test_signer] = json.load(handle)
                state["rounds"][test_signer]["status"] = "completed_existing"
                if not args.dry_run:
                    save_json_atomic(state_path, state)
                continue

            state["rounds"][test_signer]["status"] = "training"
            if not args.dry_run:
                save_json_atomic(state_path, state)

            train_command = [
                sys.executable,
                str(SCRIPTS_DIR / "train.py"),
                "--config", str(Path(args.config)),
                "--data-dir", str(Path(args.data_dir)),
                "--processed-dir", str(processed_dir),
                "--epochs", str(args.epochs),
                "--seed", str(args.seed),
                "--experiment", round_name,
                "--save-dir", str(run_dir),
            ]
            if args.batch_size is not None:
                train_command.extend(["--batch-size", str(args.batch_size)])
            if args.num_workers is not None:
                train_command.extend(["--num-workers", str(args.num_workers)])
            if args.device:
                train_command.extend(["--device", args.device])
            process_warning = None
            try:
                run_command(train_command, args.dry_run)
            except subprocess.CalledProcessError as error:
                required_artifacts = (
                    metrics_path,
                    experiment_dir / "best.pt",
                    experiment_dir / "history.json",
                )
                if not args.dry_run and all(path.exists() for path in required_artifacts):
                    process_warning = (
                        f"Processo retornou {error.returncode} após gravar todos os "
                        "artefatos obrigatórios. Resultados preservados."
                    )
                    logger.warning(process_warning)
                else:
                    raise

            if not args.dry_run:
                if not metrics_path.exists():
                    raise FileNotFoundError(f"Treino terminou sem gerar {metrics_path}")
                with open(metrics_path, encoding="utf-8") as handle:
                    results[test_signer] = json.load(handle)
                state["rounds"][test_signer]["status"] = "completed"
                if process_warning:
                    state["rounds"][test_signer]["process_warning"] = process_warning
                save_json_atomic(state_path, state)
                save_summary(run_dir, run_config, results)

        if not args.dry_run:
            state["status"] = "prepared" if args.prepare_only else "completed"
            state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            save_json_atomic(state_path, state)
            summary = save_summary(run_dir, run_config, results)
            logger.info("Execução registrada em %s", run_dir)
            if summary["aggregate"]:
                logger.info("Resumo agregado: %s", json.dumps(summary["aggregate"], ensure_ascii=False))
    except BaseException as error:
        if not args.dry_run:
            state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            state["error"] = repr(error)
            state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            save_json_atomic(state_path, state)
        raise


if __name__ == "__main__":
    main()
