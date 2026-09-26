#!/usr/bin/env python3
"""
Constrói dataset final: gera splits train/val/test e salva metadados.

Uso:
    python scripts/build_dataset.py --landmarks data/landmarks --output data/processed
"""
import argparse
from collections import Counter
import json
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Construir dataset com splits")
    parser.add_argument("--landmarks", type=str, default="data/landmarks")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument(
        "--strategy",
        type=str,
        default="stratified",
        choices=["stratified", "group", "signer_holdout"],
    )
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotations", type=Path, default=PROJECT_ROOT / "data" / "annotations.csv",
                        help="CSV original usado para selecionar as classes em tempo de execução")
    parser.add_argument("--extra-keep", nargs="*", default=[],
                        help="Classes adicionais preservadas literalmente")
    parser.add_argument(
        "--test-signer",
        type=str,
        help="Articulador reservado exclusivamente para teste em signer_holdout",
    )
    parser.add_argument(
        "--train-signers",
        nargs="+",
        help="Articuladores usados no treino; por padrão, todos exceto o teste",
    )
    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Limite de classes; falha se for menor que o vocabulário protegido",
    )
    args = parser.parse_args()

    from ml.data.split import DataSplitter, SplitConfig
    from ml.data.vocabulary import read_annotations, select_vocabulary
    landmarks_dir = Path(args.landmarks)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carregar amostras do manifesto
    manifest_path = landmarks_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"Manifesto não encontrado: {manifest_path}. Execute extract_landmarks.py primeiro.")
        return

    with open(manifest_path, encoding="utf-8") as f:
        samples = json.load(f)

    logger.info(f"Total de amostras: {len(samples)}")
    annotations = read_annotations(args.annotations)
    unknown_extra = set(args.extra_keep) - set(annotations)
    if unknown_extra:
        parser.error(f"--extra-keep contém classes ausentes do CSV: {sorted(unknown_extra)}")
    selected = select_vocabulary(set(annotations), set(args.extra_keep))
    manifest_labels = {sample["label"] for sample in samples}
    selected_labels = set(selected)
    absent = selected_labels - manifest_labels
    if absent:
        parser.error(f"Classes protegidas ausentes dos landmarks: {sorted(absent)[:12]}")
    manifest_counts = Counter((sample["label"], sample.get("signer", "unknown"))
                              for sample in samples if sample["label"] in selected_labels)
    count_mismatches = [
        (label, signer, count, manifest_counts[label, signer])
        for label in selected_labels for signer, count in annotations[label].items()
        if manifest_counts[label, signer] != count
    ]
    if count_mismatches:
        parser.error("CSV e manifesto divergem em classe/articulador: "
                     f"{count_mismatches[:6]}")
    if not selected_labels:
        parser.error("A política de vocabulário não selecionou nenhuma classe")
    if args.max_classes is not None and args.max_classes < len(selected_labels):
        parser.error(
            f"--max-classes={args.max_classes} excluiria classes protegidas "
            f"({len(selected_labels)} selecionadas). Aumente o limite ou omita a opção."
        )
    logger.info("Vocabulário protegido: %s classes de %s no CSV", len(selected), len(annotations))

    # Split
    split_config = SplitConfig(
        strategy=args.strategy,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )
    splitter = DataSplitter(split_config)

    split_metadata: dict = {}
    if args.strategy == "signer_holdout":
        if not args.test_signer:
            parser.error("--test-signer é obrigatório com --strategy signer_holdout")

        available_signers = sorted({
            str(sample.get("signer", "unknown"))
            for sample in samples
            if sample.get("signer", "unknown") != "unknown"
        })
        train_signers = args.train_signers or [
            signer for signer in available_signers if signer != args.test_signer
        ]
        required_signers = list(dict.fromkeys([*train_signers, args.test_signer]))
        covered = set(splitter.labels_with_signer_coverage(samples, required_signers))
        missing_coverage = selected_labels - covered
        if missing_coverage:
            logger.warning(
                "%s classes protegidas sem cobertura completa; permanecem no "
                "vocabulário, mas a avaliação dessas classes será incompleta: %s",
                len(missing_coverage), sorted(missing_coverage),
            )
        labels = sorted(selected_labels)
        splits = splitter.split_by_signer(
            samples,
            test_signer=args.test_signer,
            train_signers=train_signers,
            allowed_labels=set(labels),
        )
        split_metadata = {
            "train_signers": train_signers,
            "val_signers": [],
            "test_signer": args.test_signer,
            "available_signers": available_signers,
            "required_signers": required_signers,
            "max_classes": args.max_classes,
            "classes_without_full_signer_coverage": sorted(missing_coverage),
        }
    else:
        labels = sorted(selected_labels)
        eligible_indices = [i for i, sample in enumerate(samples) if sample["label"] in selected_labels]
        relative = splitter.split([samples[i] for i in eligible_indices])
        splits = {name: [eligible_indices[i] for i in indices]
                  for name, indices in relative.items()}

    label_map = {label: idx for idx, label in enumerate(labels)}
    class_counts = {
        name: dict(sorted(Counter(samples[i]["label"] for i in indices).items()))
        for name, indices in splits.items()
    }
    logger.info(f"Total de classes selecionadas: {len(label_map)}")

    # Salvar
    splits_path = output_dir / "splits.json"
    splitter.save_split(splits, splits_path)

    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    logger.info(f"Label map salvo em {label_map_path}")

    # Copiar manifesto
    shutil.copy2(manifest_path, output_dir / "manifest.json")
    vocabulary_report = {
        "annotations": str(args.annotations.resolve()),
        "included": selected,
        "excluded": sorted(set(annotations) - selected_labels),
        "extra_keep": args.extra_keep,
        "annotation_counts": {label: dict(annotations[label]) for label in labels},
    }
    with open(output_dir / "vocabulary_report.json", "w", encoding="utf-8") as f:
        json.dump(vocabulary_report, f, indent=2, ensure_ascii=False)

    # Metadados
    meta = {
        "landmarks_dir": str(landmarks_dir),
        "num_samples": len(samples),
        "num_classes": len(label_map),
        "split_strategy": args.strategy,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "class_counts_by_split": class_counts,
        "seed": args.seed,
        "selected_classes": labels,
        **split_metadata,
    }
    with open(output_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("Dataset construído com sucesso!")
    for name, idx in splits.items():
        logger.info(f"  {name}: {len(idx)} amostras")


if __name__ == "__main__":
    main()
