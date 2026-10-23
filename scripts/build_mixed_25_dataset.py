#!/usr/bin/env python3
"""Combina os 15 sinais MINDS e dez sinais novos, preservando holdout por origem."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data.split import DataSplitter, SplitConfig
OLD_CLASSES = ROOT / "configs" / "minds_libras_15_sem_cinco.json"
NEW_CLASSES = ("Água", "Ajudar", "Amigo", "Aprender", "Beber",
               "Brincar", "Cansado", "Certo", "Comer", "Diretor")
NEW_TRAIN = {f"Sinalizador{i:02d}" for i in range(1, 6)}
NEW_VAL = "Sinalizador06"
NEW_TEST = "Sinalizador07"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assemble(old_dir: Path, old_split_dir: Path,
             new_sources: list[Path], output_dir: Path) -> dict:
    old_manifest = read_json(old_split_dir / "manifest.json")
    if old_manifest != read_json(old_dir / "manifest.json"):
        raise ValueError("Manifesto MINDS diverge do split de 15 classes")
    old_splits = read_json(old_split_dir / "splits.json")
    old_labels = set(read_json(OLD_CLASSES)["classes_minds_libras_15"])
    if old_labels != set(read_json(old_split_dir / "label_map.json")):
        raise ValueError("As 15 classes antigas divergem do split arquivado")
    if len(new_sources) != 2 or len({p.resolve() for p in new_sources}) != 2:
        raise ValueError("Informe os dois diretórios distintos das dez palavras novas")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    splits = {name: [] for name in ("train", "val", "test")}
    seen_paths: set[str] = set()
    new_labels: set[str] = set()
    new_label_sources: dict[str, Path] = {}

    def add(sample: dict, source: Path, origin: str, split_name: str):
        original = source / sample["path"]
        if not original.is_file():
            raise FileNotFoundError(original)
        # Todos os diretórios de origem ficam sob data/landmarks/. O caminho
        # relativo permite reutilizar os .npz sem duplicar ~80 MB no disco.
        relative = Path("..") / source.name / sample["path"]
        key = relative.as_posix()
        if key in seen_paths:
            raise ValueError(f"Vídeo repetido no manifesto misto: {key}")
        seen_paths.add(key)
        record = dict(sample)
        record["path"] = key
        record["video_id"] = f"{origin}:{sample.get('video_id', sample['path'])}"
        record["signer"] = f"{origin}:{sample['signer']}"
        record["origin"] = origin
        splits[split_name].append(len(manifest))
        manifest.append(record)

    for split_name in splits:
        for index in old_splits[split_name]:
            sample = old_manifest[index]
            if sample["label"] not in old_labels:
                raise ValueError(f"Classe inesperada no split MINDS: {sample['label']}")
            add(sample, old_dir, "minds", split_name)

    for source in new_sources:
        for sample in read_json(source / "manifest.json"):
            label = sample["label"]
            if label not in NEW_CLASSES or label in old_labels:
                raise ValueError(f"Classe nova inesperada ou repetida: {label}")
            previous_source = new_label_sources.setdefault(label, source)
            if previous_source != source:
                raise ValueError(f"Classe nova aparece em duas fontes: {label}")
            signer = sample["signer"]
            split_name = ("train" if signer in NEW_TRAIN else
                          "val" if signer == NEW_VAL else
                          "test" if signer == NEW_TEST else None)
            if split_name is None:
                raise ValueError(f"Sinalizador novo fora do protocolo: {signer}")
            add(sample, source, "novas", split_name)
            new_labels.add(label)

    if new_labels != set(NEW_CLASSES):
        raise ValueError(f"Faltam classes novas: {sorted(set(NEW_CLASSES) - new_labels)}")
    if set(old_labels) & new_labels or len(old_labels | new_labels) != 25:
        raise ValueError("Vocabulário misto não tem exatamente 25 classes distintas")
    if len(manifest) != 670 or {name: len(indices) for name, indices in splits.items()} != {
        "train": 500, "val": 85, "test": 85
    }:
        raise ValueError("Contagens inesperadas: esperado 670 vídeos, split 500/85/85")

    splitter = DataSplitter(SplitConfig(group_key="signer"))
    splitter.verify_group_isolation(manifest, splits, group_key="signer")
    splitter.verify_group_isolation(manifest, splits, group_key="video")
    counts = {name: Counter(manifest[i]["label"] for i in indices)
              for name, indices in splits.items()}
    for name, class_counts in counts.items():
        for label in old_labels:
            if class_counts[label] != {"train": 30, "val": 5, "test": 5}[name]:
                raise ValueError(f"Contagem MINDS incorreta: {name}/{label}")
        for label in new_labels:
            if class_counts[label] != {"train": 5, "val": 1, "test": 1}[name]:
                raise ValueError(f"Contagem nova incorreta: {name}/{label}")

    label_map = {label: i for i, label in enumerate(sorted(old_labels | new_labels))}
    report = {
        "split_strategy": "mixed_signer_holdout",
        "group_key": "signer",
        "num_samples": len(manifest), "num_classes": len(label_map),
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "class_counts_by_split": {name: dict(sorted(value.items())) for name, value in counts.items()},
        "old_source": str(old_dir.resolve()),
        "old_split_source": str(old_split_dir.resolve()),
        "new_sources": [str(p.resolve()) for p in new_sources],
        "new_train_signers": sorted(NEW_TRAIN),
        "new_val_signer": NEW_VAL, "new_test_signer": NEW_TEST,
        "selected_classes": sorted(label_map),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    processed = ROOT / "data" / "processed" / output_dir.name
    processed.mkdir(parents=True, exist_ok=True)
    for name, obj in (("manifest.json", manifest), ("splits.json", splits),
                      ("label_map.json", label_map), ("dataset_meta.json", report)):
        (processed / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    with (processed / "annotations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "class", "user_id"])
        writer.writeheader()
        for sample in manifest:
            writer.writerow({"video_id": sample["video_id"], "class": sample["label"],
                             "user_id": sample["signer"]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-landmarks", type=Path, default=ROOT / "data/landmarks/minds_libras_all20")
    parser.add_argument("--old-splits", type=Path, default=ROOT / "data/processed/minds_libras_15_sem_cinco")
    parser.add_argument("--new-landmarks", type=Path, nargs=2, default=[
        ROOT / "data/landmarks/novas_5_palavras", ROOT / "data/landmarks/novas_5_restantes"])
    parser.add_argument("--output", type=Path, default=ROOT / "data/landmarks/mixed_25")
    args = parser.parse_args()
    output = args.output.resolve()
    sources = [args.old_landmarks.resolve(), *(p.resolve() for p in args.new_landmarks)]
    if output.parent != sources[0].parent or any(p.parent != output.parent for p in sources):
        parser.error("As três fontes e a saída precisam ser subpastas irmãs em data/landmarks")
    if output in sources:
        parser.error("A saída não pode sobrescrever uma fonte")
    report = assemble(sources[0], args.old_splits.resolve(), sources[1:], output)
    print(json.dumps({"classes": report["num_classes"], "videos": report["num_samples"],
                      "splits": report["split_sizes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
