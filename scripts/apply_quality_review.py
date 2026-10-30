#!/usr/bin/env python3
"""Create a derived training split from manually reviewed quality decisions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def apply(review_csv: Path, source_dir: Path, output_dir: Path) -> dict:
    source_manifest = source_dir / "manifest.json"
    source_splits = source_dir / "splits.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    splits = json.loads(source_splits.read_text(encoding="utf-8"))
    label_map = json.loads((source_dir / "label_map.json").read_text(encoding="utf-8"))
    with review_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_path = {row["path"]: row for row in rows}
    if len(by_path) != len(rows):
        raise ValueError("Caminho duplicado na revisão")
    indices = {sample["path"]: i for i, sample in enumerate(manifest)}
    train_set = set(splits["train"])
    excluded = []
    for path, row in by_path.items():
        if path not in indices:
            raise ValueError(f"Vídeo ausente do manifesto congelado: {path}")
        decision = row.get("review_decision", "").strip().lower()
        if decision in {"pending", "not_flagged", "keep", ""}:
            continue
        if decision not in {"exclude_train", "rerecord", "reextract"}:
            raise ValueError(f"Decisão inválida: {decision}")
        if not row.get("reviewer", "").strip() or not row.get("review_reason", "").strip():
            raise ValueError(f"Revisão sem pessoa/motivo: {path}")
        index = indices[path]
        if index not in train_set:
            raise ValueError("Não excluir validação/teste depois de medir; crie outro protocolo")
        if decision != "exclude_train":
            raise ValueError(f"{decision} requer novo arquivo e reextração antes de gerar split")
        excluded.append(index)
    remaining = [i for i in splits["train"] if i not in set(excluded)]
    counts = Counter(manifest[i]["label"] for i in remaining)
    if set(counts) != set(label_map):
        raise ValueError("Exclusão deixaria uma classe sem treino")
    output_dir.mkdir(parents=True, exist_ok=True)
    new_splits = {**splits, "train": remaining}
    for name, data in (("manifest.json", manifest), ("splits.json", new_splits),
                       ("label_map.json", label_map)):
        (output_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"excluded_train": len(excluded), "remaining_train": len(remaining),
              "val": len(splits["val"]), "test": len(splits["test"]),
              "source_split_sha256": hashlib.sha256(source_splits.read_bytes()).hexdigest(),
              "review_csv_sha256": hashlib.sha256(review_csv.read_bytes()).hexdigest(),
              "excluded_paths": [manifest[i]["path"] for i in excluded]}
    (output_dir / "quality_review_applied.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("data/processed/mixed_25"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/mixed_25_quality_reviewed"))
    args = parser.parse_args()
    print(json.dumps(apply(args.review_csv, args.source_dir, args.output_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
