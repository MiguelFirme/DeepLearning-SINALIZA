#!/usr/bin/env python3
"""Adiciona negativos reais revisados ao split congelado de 25 sinais."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


NEGATIVE_LABEL = "__nenhum_sinal__"
REQUIRED = {"path", "person_id", "session_id", "subtype", "split", "reviewed"}
SPLITS = {"train", "val", "test"}
SUBTYPES = {"repouso", "conversa", "gesto_incompleto", "movimento_fora_vocabulario"}


def prepare(positive_dir: Path, positive_split_dir: Path,
            raw_negative_dir: Path, negative_landmarks_dir: Path,
            review_csv: Path, output_dir: Path, processed_dir: Path) -> dict:
    positive_dir = positive_dir.resolve()
    output_dir = output_dir.resolve()
    negative_landmarks_dir = negative_landmarks_dir.resolve()
    if not (positive_dir.parent == output_dir.parent == negative_landmarks_dir.parent):
        raise ValueError("Saída, positivos e negativos precisam estar sob data/landmarks")
    if output_dir in (positive_dir, negative_landmarks_dir):
        raise ValueError("Saída não pode sobrescrever uma origem")
    manifest = json.loads((positive_dir / "manifest.json").read_text(encoding="utf-8"))
    archived = json.loads((positive_split_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest != archived or len(manifest) != 670:
        raise ValueError("Base positiva de 25 classes mudou; reconstrua ou audite antes")
    splits = json.loads((positive_split_dir / "splits.json").read_text(encoding="utf-8"))
    label_map = json.loads((positive_split_dir / "label_map.json").read_text(encoding="utf-8"))
    if len(label_map) != 25 or NEGATIVE_LABEL in label_map:
        raise ValueError("Mapa positivo precisa conter somente as 25 classes aprovadas")
    split_of_person = {}
    for split, indices in splits.items():
        for index in indices:
            person = manifest[index]["signer"]
            previous = split_of_person.setdefault(person, split)
            if previous != split:
                raise ValueError(f"Pessoa positiva em splits distintos: {person}")

    with review_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV de revisão requer {sorted(REQUIRED)}")
        reviewed = list(reader)
    if not reviewed:
        raise ValueError("Nenhum negativo revisado; grave e revise vídeos antes de treinar")

    session_owner = {}
    seen = set()
    counts = Counter()
    for row in reviewed:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".mp4":
            raise ValueError(f"Caminho de vídeo inválido: {path}")
        if row["reviewed"].strip().lower() != "yes":
            raise ValueError(f"Vídeo sem revisão aprovada: {path}")
        split = row["split"].strip()
        if split not in SPLITS or row["subtype"].strip() not in SUBTYPES:
            raise ValueError(f"Split ou subtipo inválido: {path}")
        person = row["person_id"].strip()
        session = row["session_id"].strip()
        if not person or ":" not in person or not session:
            raise ValueError(f"Identifique pessoa como origem:id e sessão: {path}")
        previous = split_of_person.setdefault(person, split)
        if previous != split:
            raise ValueError(f"Pessoa {person} aparece em {previous} e {split}")
        session_key = (person, session)
        session_previous = session_owner.setdefault(session_key, split)
        if session_previous != split:
            raise ValueError(f"Sessão {session_key} vazou entre splits")
        if path.as_posix() in seen:
            raise ValueError(f"Vídeo negativo repetido: {path}")
        seen.add(path.as_posix())
        video = raw_negative_dir / path
        npz = negative_landmarks_dir / path.with_suffix(".npz")
        if not video.is_file() or not npz.is_file():
            raise FileNotFoundError(f"Vídeo/landmarks ausentes: {video} / {npz}")
        with np.load(npz) as data:
            if data["landmarks"].ndim != 2 or data["landmarks"].shape[1] != 346:
                raise ValueError(f"Landmarks inválidos: {npz}")
            if data["mask"].shape != (len(data["landmarks"]), 4):
                raise ValueError(f"Máscara inválida: {npz}")
            if len(data["landmarks"]) < 24:
                raise ValueError(f"Negativo curto demais (<24 quadros): {npz}")
        sample = {
            "path": (Path("..") / negative_landmarks_dir.name / path.with_suffix(".npz")).as_posix(),
            "video_id": f"negatives:{path.as_posix()}", "label": NEGATIVE_LABEL,
            "signer": person, "session": f"{person}:{session}",
            "negative_subtype": row["subtype"].strip(), "reviewed": True,
        }
        splits[split].append(len(manifest))
        manifest.append(sample)
        counts[split] += 1
    minimum = {"train": 20, "val": 5, "test": 5}
    if any(counts[split] < minimum[split] for split in SPLITS):
        raise ValueError(f"Negativos insuficientes por split: {dict(counts)}; mínimo {minimum}")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    label_map[NEGATIVE_LABEL] = 25
    meta = {"split_strategy": "reviewed_negative_person_holdout",
            "positive_source": str(positive_dir), "negative_source": str(negative_landmarks_dir),
            "negative_counts": dict(counts), "num_classes": 26,
            "split_sizes": {name: len(indices) for name, indices in splits.items()},
            "review_csv": str(review_csv),
            "warning": "Negativos foram revisados; teste positivo anterior continua exploratório."}
    for target, objects in ((output_dir / "manifest.json", manifest),
                            (processed_dir / "manifest.json", manifest),
                            (processed_dir / "splits.json", splits),
                            (processed_dir / "label_map.json", label_map),
                            (processed_dir / "dataset_meta.json", meta)):
        target.write_text(json.dumps(objects, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-dir", type=Path, default=Path("data/landmarks/mixed_25"))
    parser.add_argument("--positive-split-dir", type=Path, default=Path("data/processed/mixed_25"))
    parser.add_argument("--raw-negative-dir", type=Path, default=Path("data/raw/negatives"))
    parser.add_argument("--negative-landmarks-dir", type=Path, default=Path("data/landmarks/negatives"))
    parser.add_argument("--review-csv", type=Path, default=Path("data/negative_examples_reviewed.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/landmarks/mixed25_rejection"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/mixed25_rejection"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.positive_dir, args.positive_split_dir,
                             args.raw_negative_dir, args.negative_landmarks_dir,
                             args.review_csv, args.output_dir, args.processed_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
