#!/usr/bin/env python3
"""Build an isolated 26-class pilot from unreviewed boundary candidates.

Weak labels: this output must not replace the reviewed-negative dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


NEGATIVE_LABEL = "__nenhum_sinal__"


def build(candidates_csv: Path, positive_dir: Path, source_processed: Path,
          segment_dir: Path, output_dir: Path, output_processed: Path,
          allow_weak_labels: bool = False) -> dict:
    if not allow_weak_labels:
        raise ValueError("Piloto com rótulos fracos exige --allow-weak-labels explícito")
    positive_dir = positive_dir.resolve()
    segment_dir = segment_dir.resolve()
    output_dir = output_dir.resolve()
    if len({positive_dir, segment_dir, output_dir}) != 3:
        raise ValueError("Origens e saída precisam ser diretórios distintos")
    if not (positive_dir.parent == segment_dir.parent == output_dir.parent):
        raise ValueError("Todos os landmarks precisam estar sob o mesmo diretório pai")
    manifest = json.loads((positive_dir / "manifest.json").read_text(encoding="utf-8"))
    archived = json.loads((source_processed / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((source_processed / "splits.json").read_text(encoding="utf-8"))
    label_map = json.loads((source_processed / "label_map.json").read_text(encoding="utf-8"))
    if manifest != archived or len(manifest) != 670 or len(label_map) != 25:
        raise ValueError("Esperada a base original de 670 vídeos e 25 classes")
    owners = {i: split for split in splits for i in splits[split]}
    with candidates_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Sem candidatos de repouso")
    if any(row["split"] == "test" for row in rows):
        raise ValueError("O piloto não pode consultar candidatos de teste")
    seen = set()
    selected = []
    for row in rows:
        index = int(row["manifest_index"])
        start, end = int(row["start_frame"]), int(row["end_frame_exclusive"])
        key = (index, start, end)
        if key in seen:
            raise ValueError(f"Intervalo duplicado: {key}")
        seen.add(key)
        sample = manifest[index]
        if row["split"] not in {"train", "val"} or owners[index] != row["split"]:
            raise ValueError("Candidato mudou de split")
        if row["signer"] != sample["signer"] or row["source_npz"] != sample["path"]:
            raise ValueError("Candidato não corresponde ao vídeo de origem")
        if row["review_decision"] not in {"pending", "approved"}:
            continue
        with np.load(positive_dir / sample["path"]) as data:
            length = len(data["landmarks"])
            if end - start != 32 or start < 0 or end > length:
                raise ValueError("Piloto exige trecho original de 32 quadros")
            if start != 0 and end != length:
                raise ValueError("Trecho não está na borda do vídeo")
            landmarks = data["landmarks"][start:end].copy()
            mask = data["mask"][start:end].copy()
        if landmarks.shape != (32, 346) or mask.shape != (32, 4):
            raise ValueError("Landmarks/máscara incompatíveis")
        selected.append((row, sample, landmarks, mask))
    counts = Counter(row["split"] for row, *_ in selected)
    if counts["train"] < 20 or counts["val"] < 5:
        raise ValueError(f"Quantidade insuficiente para piloto: {dict(counts)}")
    segment_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_processed.mkdir(parents=True, exist_ok=True)
    for row, sample, landmarks, mask in selected:
        stem = f"{int(row['manifest_index']):03d}_{row['edge']}"
        target = segment_dir / f"{stem}.npz"
        np.savez_compressed(target, landmarks=landmarks, mask=mask,
                            source_video=sample["video_id"],
                            start_frame=int(row["start_frame"]),
                            end_frame_exclusive=int(row["end_frame_exclusive"]))
        manifest.append({"path": f"../{segment_dir.name}/{stem}.npz",
                         "video_id": f"edge:{sample['video_id']}:{stem}",
                         "label": NEGATIVE_LABEL, "signer": sample["signer"],
                         "source_label": sample["label"], "source_manifest_index": int(row["manifest_index"]),
                         "edge": row["edge"], "weak_label": True})
        splits[row["split"]].append(len(manifest) - 1)
    label_map[NEGATIVE_LABEL] = 25
    meta = {"pilot": True, "weak_labels": True, "review_status": "AI screened contact sheets; no human full-clip review",
            "candidate_csv_sha256": hashlib.sha256(candidates_csv.read_bytes()).hexdigest(),
            "negative_counts": dict(counts), "split_sizes": {k: len(v) for k, v in splits.items()},
            "test_negatives": 0, "segment_frames": 32,
            "warning": "Not suitable for deployment or calibrated rejection."}
    for target, data in ((output_dir / "manifest.json", manifest),
                         (output_processed / "manifest.json", manifest),
                         (output_processed / "splits.json", splits),
                         (output_processed / "label_map.json", label_map),
                         (output_processed / "dataset_meta.json", meta)):
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("reports/idle_edges25/candidates.csv"))
    parser.add_argument("--positive-dir", type=Path, default=Path("data/landmarks/mixed_25"))
    parser.add_argument("--source-processed", type=Path, default=Path("data/processed/mixed_25"))
    parser.add_argument("--segment-dir", type=Path, default=Path("data/landmarks/idle_edges32"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/landmarks/mixed25_edge32_pilot"))
    parser.add_argument("--output-processed", type=Path, default=Path("data/processed/mixed25_edge32_pilot"))
    parser.add_argument("--allow-weak-labels", action="store_true")
    args = parser.parse_args()
    result = build(args.candidates, args.positive_dir, args.source_processed,
                   args.segment_dir, args.output_dir, args.output_processed,
                   args.allow_weak_labels)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
