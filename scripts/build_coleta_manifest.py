#!/usr/bin/env python3
"""Cria manifesto de coleta e splits explícitos por pessoa, sem copiar .npz."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data.dataset_v2 import read_metadata
from ml.training.pipeline_v2 import validate_splits_v2


def build_collection_manifest(source, output, processed, train_people, val_people, test_people,
                              merge_manifest=None, person_map=None, force=False):
    """Nomes canônicos de pessoa precisam ser consistentes entre todas as origens."""
    source, output, processed = Path(source).resolve(), Path(output).resolve(), Path(processed).resolve()
    people = {"train": set(train_people), "val": set(val_people), "test": set(test_people)}
    if any(people[a] & people[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Uma pessoa não pode pertencer a dois splits")
    person_map = person_map or {}
    rows = []
    for path in sorted(source.rglob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            meta = read_metadata(data)
            raw, mask, times = data["landmarks"], data["mask"], data["timestamps"]
            if (len(raw) < 2 or raw.shape != (len(times), 346) or mask.shape != (len(times), 4)
                    or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                    or not np.isfinite(raw).all()):
                raise ValueError(f"Arquivo de coleta inválido: {path}")
        if meta.get("consent") is not True or not meta.get("person") or not meta.get("word"):
            raise ValueError(f"Metadados de pessoa/rótulo/consentimento ausentes: {path}")
        if meta.get("width", 0) <= 0 or meta.get("height", 0) <= 0:
            raise ValueError(f"Dimensões ausentes na coleta: {path}")
        person = person_map.get(meta["person"], meta["person"])
        rows.append({"path": os.path.relpath(path, output).replace("\\", "/"), "label": meta["word"],
                     "signer": person, "person": person, "session": meta.get("session"),
                     "origin": "novas", "video_id": str(path), "source": "coleta"})
    if merge_manifest:
        merge_path = Path(merge_manifest).resolve()
        for original in json.loads(merge_path.read_text(encoding="utf-8")):
            row = original.copy()
            original_path = (merge_path.parent / row["path"]).resolve()
            row["path"] = os.path.relpath(original_path, output).replace("\\", "/")
            person = row.get("person") or row.get("signer", "unknown")
            row["person"] = row["signer"] = person_map.get(person, person)
            rows.append(row)
    if not rows or len({str((output / row["path"]).resolve()) for row in rows}) != len(rows):
        raise ValueError("Coleta vazia ou arquivos duplicados na mescla")
    known_people = {row["signer"] for row in rows}
    assigned = set.union(*people.values())
    if known_people != assigned:
        raise ValueError(f"Informe todas e somente as pessoas; faltam {known_people - assigned}, sobram {assigned - known_people}")
    labels = {word: i for i, word in enumerate(sorted({row["label"] for row in rows}))}
    splits = {split: [i for i, row in enumerate(rows) if row["signer"] in ids] for split, ids in people.items()}
    validate_splits_v2(rows, splits, labels)
    if not splits["test"]:
        raise ValueError("Teste por pessoa não pode ficar vazio")
    files = [(output / "manifest.json", rows), (processed / "manifest.json", rows),
             (processed / "label_map.json", labels), (processed / "splits.json", splits),
             (processed / "dataset_meta.json", {"group_key": "signer", "split_strategy": "group", "persons": {k: sorted(v) for k, v in people.items()}})]
    if not force and any(path.exists() for path, _ in files):
        raise ValueError("Saída já existe; use outro diretório ou --force conscientemente")
    for path, value in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"samples": len(rows), "classes": len(labels), "splits": {key: len(value) for key, value in splits.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/coleta")
    parser.add_argument("--output", default="data/landmarks/coleta_v2")
    parser.add_argument("--processed-dir", default="data/processed/coleta_v2")
    parser.add_argument("--train-persons", nargs="+", required=True)
    parser.add_argument("--val-persons", nargs="+", required=True)
    parser.add_argument("--test-persons", nargs="+", required=True)
    parser.add_argument("--merge-manifest")
    parser.add_argument("--person-map", help="JSON que mapeia IDs antigos e novos para a mesma pessoa real")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    mapping = json.loads(Path(args.person_map).read_text(encoding="utf-8")) if args.person_map else None
    print(json.dumps(build_collection_manifest(args.input, args.output, args.processed_dir, args.train_persons,
                                               args.val_persons, args.test_persons, args.merge_manifest, mapping, args.force), indent=2))


if __name__ == "__main__":
    main()
