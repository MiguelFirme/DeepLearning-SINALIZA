#!/usr/bin/env python3
"""Filtra os índices de 41 classes preservando exatamente os splits por pessoa."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"Cinco", "Vacina", "Amarelo"}


def filter_indices(manifest, splits):
    """Remapeia somente os índices retidos; não redistribui pessoas ou exemplos."""
    kept = [i for i, row in enumerate(manifest) if row["label"] not in EXCLUDED]
    mapping = {old: new for new, old in enumerate(kept)}
    rows = [manifest[i] for i in kept]
    filtered = {name: [mapping[i] for i in ids if i in mapping] for name, ids in splits.items()}
    return rows, filtered


def main():
    source = ROOT / "data/processed/mixed_grupo_20261117"
    original_data = ROOT / "data/landmarks/mixed_grupo_20261117"
    destination = ROOT / "data/landmarks/mixed_grupo39_20261120"
    processed = ROOT / "data/processed/mixed_grupo39_20261120"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest != json.loads((original_data / "manifest.json").read_text(encoding="utf-8")):
        raise ValueError("Manifestos de origem divergentes")
    splits = json.loads((source / "splits.json").read_text(encoding="utf-8"))
    rows, filtered = filter_indices(manifest, splits)
    labels = {label: i for i, label in enumerate(sorted({r["label"] for r in rows}))}
    if len(labels) != 39 or len(rows) != 635:
        raise ValueError("Origem diferente do conjunto esperado de 715 vídeos/41 classes")
    # As duas pastas ficam no mesmo nível, preservando os caminhos ../ dos .npz.
    for row in rows:
        old, new = original_data / row["path"], destination / row["path"]
        if old.resolve() != new.resolve() or not old.is_file():
            raise ValueError(f"Landmark ausente ou caminho incompatível: {new}")
    all_ids = [i for ids in filtered.values() for i in ids]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(range(len(rows))):
        raise ValueError("Os splits não formam uma partição dos exemplos")
    signers = {s: {rows[i]["signer"] for i in ids} for s, ids in filtered.items()}
    for a in signers:
        for b in signers:
            if a != b and signers[a] & signers[b]:
                raise ValueError("Pessoa presente em mais de um split")
    meta = json.loads((source / "dataset_meta.json").read_text(encoding="utf-8"))
    meta.update(num_samples=len(rows), num_classes=len(labels), selected_classes=list(labels),
                split_sizes={s: len(ids) for s, ids in filtered.items()},
                class_counts_by_split={s: dict(Counter(rows[i]["label"] for i in ids))
                                       for s, ids in filtered.items()},
                signers_by_split={s: sorted(ids) for s, ids in signers.items()},
                excluded_classes=sorted(EXCLUDED),
                exclusions_absent_in_source=sorted(EXCLUDED - {r["label"] for r in manifest}),
                parent_manifest_sha256=hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest())
    values = {"manifest.json": rows, "splits.json": filtered,
              "label_map.json": labels, "dataset_meta.json": meta}
    # Não sobrescreve silenciosamente um conjunto previamente gerado diferente.
    for folder in (destination, processed):
        for filename, value in values.items():
            path = folder / filename
            if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
                raise ValueError(f"Índice existente divergente: {path}")
    for folder in (destination, processed):
        folder.mkdir(parents=True, exist_ok=True)
        for filename, value in values.items():
            (folder / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"num_classes": len(labels), "num_samples": len(rows),
                      "split_sizes": meta["split_sizes"],
                      "already_absent": meta["exclusions_absent_in_source"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
