"""Validação cruzada v2: uma pessoa no teste e outra na validação."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from ml.training.pipeline_v2 import validate_splits_v2


def make_person_folds(manifest, people=None):
    """Nenhuma identidade aparece em mais de um split do mesmo fold."""
    people = people or sorted({row.get("person") or row.get("signer", "unknown") for row in manifest})
    known = {row.get("person") or row.get("signer") for row in manifest}
    if len(people) < 3 or len(set(people)) != len(people) or not set(people) <= known or "unknown" in people:
        raise ValueError("São necessárias pelo menos três pessoas identificadas")
    folds = []
    for index, person in enumerate(people):
        validation = people[(index + 1) % len(people)]
        splits = {"train": [], "val": [], "test": []}
        for i, row in enumerate(manifest):
            who = row.get("person") or row.get("signer")
            if who not in people:
                continue
            split = "test" if who == person else "val" if who == validation else "train"
            splits[split].append(i)
        folds.append((person, splits))
    return folds


def run_cv_v2(args):
    """Usa o CLI comum de treino e registra também métricas por origem/matriz."""
    from scripts.run_signer_cv import save_json_atomic, aggregate_results
    data_dir = Path(args.data_dir).resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    words = sorted({row["label"] for row in manifest})
    if args.essential_categories or args.extra_keep or args.class_list:
        raise ValueError("V2: use um manifesto já selecionado para o vocabulário desejado")
    if args.max_classes:
        words = words[:args.max_classes]
    # Conserva os índices do manifesto de origem.
    selected = {word: i for i, word in enumerate(words)}
    name = args.run_name or f"transformer_v2_signer_cv_seed{args.seed}"
    output = Path(args.run_root) / name
    processed_root = Path(args.processed_root) / name
    folds = make_person_folds(manifest, args.signers)
    if args.round != "all" and not 1 <= int(args.round) <= len(folds):
        raise ValueError("Número de fold fora do intervalo de pessoas")
    results = {}
    for number, (person, splits) in enumerate(folds, 1):
        if args.round != "all" and number != int(args.round):
            continue
        splits = {key: [i for i in values if manifest[i]["label"] in selected] for key, values in splits.items()}
        validate_splits_v2(manifest, splits, selected)
        tag = f"pessoa_{number:02d}"
        processed = processed_root / tag
        if not args.dry_run:
            for filename, value in (("manifest.json", manifest), ("splits.json", splits),
                                    ("label_map.json", selected), ("dataset_meta.json", {"group_key": "signer", "test_person": person})):
                save_json_atomic(processed / filename, value)
        metrics_path = output / tag / "test_metrics.json"
        if args.prepare_only:
            continue
        command = [sys.executable, "scripts/train.py", "--config", args.config, "--data-dir", str(data_dir),
                   "--processed-dir", str(processed), "--save-dir", str(output), "--experiment", tag,
                   "--seed", str(args.seed), "--epochs", str(args.epochs)]
        for option, value in (("--device", args.device), ("--num-workers", args.num_workers), ("--batch-size", args.batch_size)):
            if value is not None:
                command.extend((option, str(value)))
        print(f"Pessoa de teste: {person}; comando: {subprocess.list2cmdline(command)}")
        if args.dry_run:
            continue
        if args.force or not metrics_path.exists():
            subprocess.run(command, check=True)
        results[person] = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(json.dumps({"pessoa": person, "metricas": results[person]}, ensure_ascii=False))
        save_json_atomic(output / "summary.json", {"rounds": results, "aggregate": aggregate_results(results)})
