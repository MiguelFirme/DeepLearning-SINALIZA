#!/usr/bin/env python3
"""Combina 13 classes MINDS e dez do grupo, com holdout por pessoa/origem."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.training.pipeline_v2 import validate_splits_v2

MINDS_CLASSES = {"Acontecer", "Aluno", "Amarelo", "Aproveitar", "Banco", "Banheiro", "Barulho",
                 "Conhecer", "Filho", "Medo", "Ruim", "Vacina", "Vontade"}
GROUP_CLASSES = {"Água", "Ajudar", "Amigo", "Aprender", "Beber", "Brincar", "Cansado", "Certo", "Comer", "Diretor"}
PROTOCOL = {
    "minds": {"train": {"Sinalizador01", "Sinalizador02", "Sinalizador05", "Sinalizador06", "Sinalizador08", "Sinalizador10"},
              "val": {"Sinalizador11"}, "test": {"Sinalizador12"}},
    "novas": {"train": {f"Sinalizador{i:02d}" for i in range(1, 6)},
              "val": {"Sinalizador06"}, "test": {"Sinalizador07"}},
}


def assemble_mixed23(minds_dir, group_dir, output_dir, processed_dir):
    """Reutiliza os NPZ originais por caminho relativo, sem copiar sequências."""
    output_dir, processed_dir = Path(output_dir).resolve(), Path(processed_dir).resolve()
    manifest, splits = [], {"train": [], "val": [], "test": []}
    seen = set()
    for origin, source, expected in (("minds", Path(minds_dir).resolve(), MINDS_CLASSES),
                                      ("novas", Path(group_dir).resolve(), GROUP_CLASSES)):
        if output_dir == source or processed_dir == source or output_dir == processed_dir:
            raise ValueError("As saídas precisam ser distintas das fontes e entre si")
        path = source / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Falta {path}; conclua a extração dessa origem primeiro")
        samples = json.loads(path.read_text(encoding="utf-8"))
        if {row["label"] for row in samples} != expected:
            raise ValueError(f"Vocabulário {origin} diferente do esperado. Verifique as classes e os manifestos")
        for sample in samples:
            original = (source / sample["path"]).resolve()
            if not original.is_file():
                raise FileNotFoundError(original)
            if original in seen:
                raise ValueError(f"Arquivo duplicado: {original}")
            seen.add(original)
            who = sample["signer"]
            possible = [split for split, people in PROTOCOL[origin].items() if who in people]
            if len(possible) != 1:
                raise ValueError(f"Pessoa fora do protocolo: {origin}/{who}")
            split = possible[0]
            row = dict(sample)
            row.update(path=os.path.relpath(original, output_dir).replace("\\", "/"), origin=origin,
                       signer=f"{origin}:{who}", person=f"{origin}:{who}",
                       video_id=f"{origin}:{sample.get('video_id', sample['path'])}")
            splits[split].append(len(manifest))
            manifest.append(row)
    labels = {name: i for i, name in enumerate(sorted(MINDS_CLASSES | GROUP_CLASSES))}
    validate_splits_v2(manifest, splits, labels)
    counts = {split: Counter(manifest[i]["label"] for i in indices) for split, indices in splits.items()}
    for split in splits:
        for label in labels:
            expected = ({"train": 30, "val": 5, "test": 5} if label in MINDS_CLASSES else
                        {"train": 5, "val": 1, "test": 1})[split]
            if counts[split][label] != expected:
                raise ValueError(f"Contagem inesperada em {split}/{label}: {counts[split][label]}, esperado {expected}")
    report = {"split_strategy": "mixed_signer_holdout", "group_key": "signer", "num_samples": len(manifest),
              "num_classes": len(labels), "split_sizes": {key: len(value) for key, value in splits.items()},
              "class_counts_by_split": {key: dict(value) for key, value in counts.items()},
              "signers": {origin: {key: sorted(value) for key, value in groups.items()} for origin, groups in PROTOCOL.items()}}
    files = [(output_dir / "manifest.json", manifest), (processed_dir / "manifest.json", manifest),
             (processed_dir / "label_map.json", labels), (processed_dir / "splits.json", splits),
             (processed_dir / "dataset_meta.json", report)]
    # Permite repetir o comando somente se o protocolo existente for idêntico.
    for path, value in files:
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Saída já existe com protocolo diferente; use outro diretório: {path}")
    for path, value in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minds-landmarks", type=Path, default=Path("data/landmarks/minds_libras_all20"))
    parser.add_argument("--group-landmarks", type=Path, default=Path("data/landmarks/videos_do_grupo"))
    parser.add_argument("--output", type=Path, default=Path("data/landmarks/mixed_23"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/mixed_23"))
    args = parser.parse_args()
    try:
        report = assemble_mixed23(args.minds_landmarks, args.group_landmarks, args.output, args.processed_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Não foi possível montar o dataset: {exc}\n")
    print(json.dumps({key: report[key] for key in ("num_samples", "num_classes", "split_sizes")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
