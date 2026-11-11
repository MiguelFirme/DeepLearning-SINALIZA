#!/usr/bin/env python3
"""Verifica landmarks e restaura apenas índices arquivados do mixed25."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]


def restore_dataset(archive, landmarks_root, processed_dir, check_only=False):
    """Não recria landmarks nem sobrescreve um protocolo local diferente."""
    archive, landmarks_root, processed_dir = Path(archive), Path(landmarks_root), Path(processed_dir)
    destination = landmarks_root / "mixed_25"
    names = ("manifest.json", "splits.json", "label_map.json", "dataset_meta.json")
    for name in names:
        if not (archive / name).is_file():
            raise FileNotFoundError(f"Índice arquivado ausente: {archive / name}")
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((archive / "splits.json").read_text(encoding="utf-8"))
    labels = json.loads((archive / "label_map.json").read_text(encoding="utf-8"))
    if len(manifest) != 670 or len(labels) != 25 or {key: len(value) for key, value in splits.items()} != {"train": 500, "val": 85, "test": 85}:
        raise ValueError("O arquivo não corresponde ao protocolo mixed25 de 670 vídeos")
    missing = [row for row in manifest if not (destination / row["path"]).is_file()]
    report = {"expected": len(manifest), "found": len(manifest) - len(missing), "missing": len(missing),
              "missing_by_source": dict(Counter(Path(row["path"]).parts[1] for row in missing)),
              "examples": [str(destination / row["path"]) for row in missing[:3]], "restored": False}
    if missing or check_only:
        return report
    targets = [(destination / "manifest.json", archive / "manifest.json")]
    targets += [(processed_dir / name, archive / name) for name in names]
    for target, source in targets:
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Índice local diferente; preservado sem sobrescrever: {target}")
    for target, source in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    report["restored"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "Modelos salvos/transformer_mixed25_full/data_split")
    parser.add_argument("--landmarks-root", type=Path, default=ROOT / "data/landmarks")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed/mixed_25")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        report = restore_dataset(args.archive, args.landmarks_root, args.processed_dir, args.check_only)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Não foi possível restaurar os índices: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["missing"]:
        parser.exit(2, "Faltam landmarks de treino. Recupere os .npz das pastas indicadas; os pesos best.pt e o annotations.csv não contêm essas sequências. Nenhum índice foi criado.\n")


if __name__ == "__main__":
    main()
