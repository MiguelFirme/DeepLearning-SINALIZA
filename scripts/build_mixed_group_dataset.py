#!/usr/bin/env python3
"""Combina MINDS selecionado com todas as classes atuais dos vídeos do grupo."""
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
from scripts.build_mixed_23_dataset import MINDS_CLASSES, PROTOCOL


def assemble(minds_dir, group_dir, group_videos, output, processed):
    """Descobre vocabulário e conserva identidade, sem completar amostras faltantes."""
    minds_dir, group_dir, group_videos, output, processed = map(
        lambda path: Path(path).resolve(), (minds_dir, group_dir, group_videos, output, processed))
    if output == processed or output in {minds_dir, group_dir} or processed in {minds_dir, group_dir}:
        raise ValueError("As saídas precisam ser distintas entre si e das fontes")
    if not group_videos.is_dir():
        raise FileNotFoundError(group_videos)
    raw_paths = {path.relative_to(group_videos).with_suffix(".npz").as_posix()
                 for path in group_videos.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"}
    if not raw_paths:
        raise ValueError("Pasta dos vídeos do grupo vazia")
    manifest, splits, seen_files, seen_videos = [], {"train": [], "val": [], "test": []}, set(), set()
    for origin, source in (("minds", minds_dir), ("novas", group_dir)):
        path = source / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Falta {path}; conclua a extração dos landmarks primeiro")
        samples = json.loads(path.read_text(encoding="utf-8"))
        if not samples:
            raise ValueError(f"Manifesto vazio: {path}")
        if origin == "minds" and {row["label"] for row in samples} != MINDS_CLASSES:
            raise ValueError("MINDS precisa conter somente as 13 classes selecionadas")
        if origin == "novas" and {Path(row["path"]).as_posix() for row in samples} != raw_paths:
            missing = raw_paths - {Path(row["path"]).as_posix() for row in samples}
            raise ValueError(f"Manifesto do grupo não corresponde aos vídeos atuais. Faltam {len(missing)} extrações; verifique arquivos antigos ou erros")
        for sample in sorted(samples, key=lambda row: row["path"]):
            original = (source / sample["path"]).resolve()
            if not original.is_file():
                raise FileNotFoundError(original)
            video_id = f"{origin}:{sample.get('video_id', sample['path'])}"
            if original in seen_files or video_id in seen_videos:
                raise ValueError(f"Arquivo/vídeo duplicado: {original}")
            seen_files.add(original)
            seen_videos.add(video_id)
            who = sample.get("signer", "unknown")
            groups = [name for name, people in PROTOCOL[origin].items() if who in people]
            if len(groups) != 1:
                raise ValueError(f"Pessoa fora do protocolo: {origin}/{who}; informe a identidade correta antes de dividir")
            name = groups[0]
            row = dict(sample)
            row.update(path=os.path.relpath(original, output).replace("\\", "/"), origin=origin,
                       signer=f"{origin}:{who}", person=f"{origin}:{who}", video_id=video_id)
            splits[name].append(len(manifest))
            manifest.append(row)
    labels = {name: i for i, name in enumerate(sorted({row["label"] for row in manifest}))}
    validate_splits_v2(manifest, splits, labels)
    for name, indices in splits.items():
        absent = set(labels) - {manifest[i]["label"] for i in indices}
        if absent:
            raise ValueError(f"Classes sem exemplo em {name}: {sorted(absent)}; não atribua pessoas automaticamente")
    counts = {name: dict(Counter(manifest[i]["label"] for i in indices)) for name, indices in splits.items()}
    report = {"num_samples": len(manifest), "num_classes": len(labels), "selected_classes": sorted(labels),
              "split_strategy": "mixed_signer_holdout", "group_key": "signer",
              "split_sizes": {name: len(indices) for name, indices in splits.items()},
              "class_counts_by_split": counts,
              "signers_by_split": {name: sorted({manifest[i]["signer"] for i in indices}) for name, indices in splits.items()},
              "sources": {"minds": str(minds_dir), "novas": str(group_dir), "group_videos": str(group_videos)}}
    files = [(output / "manifest.json", manifest), (processed / "manifest.json", manifest),
             (processed / "label_map.json", labels), (processed / "splits.json", splits),
             (processed / "dataset_meta.json", report)]
    for path, value in files:
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Saída existente com outros dados: {path}. Use novos diretórios")
    for path, value in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minds-landmarks", type=Path, default=Path("data/landmarks/minds_libras_all20"))
    parser.add_argument("--group-landmarks", type=Path, required=True)
    parser.add_argument("--group-videos", type=Path, default=Path("data/raw/Videos do grupo"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = assemble(args.minds_landmarks, args.group_landmarks, args.group_videos, args.output, args.processed_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Não foi possível montar o dataset ampliado: {exc}\n")
    print(json.dumps({key: report[key] for key in ("num_samples", "num_classes", "split_sizes")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
