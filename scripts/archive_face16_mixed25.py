#!/usr/bin/env python3
"""Archive the two face16 checkpoints and their frozen comparison protocol."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml.features.feature_modes import FACE16_MEDIAPIPE_INDICES


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main():
    comparison_path = ROOT / "reports/fusion_mixed25_face16/comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    baseline_spec = json.loads((ROOT / "configs/fusion25_frozen.json").read_text(encoding="utf-8"))
    config_chain = (
        "default.yaml", "bilstm_regularized_school.yaml", "bilstm_minds_pose_face.yaml",
        "transformer_minds_pose_face.yaml", "transformer_mixed25_pose_face.yaml",
        "transformer_mixed25_full.yaml", "transformer_mixed25_pose_face16.yaml",
        "transformer_mixed25_full_face16.yaml",
    )
    paths = {}
    for name, mode in (("pose_face", "pose_face16"), ("full", "full_face16")):
        experiment = f"transformer_mixed25_{mode}_seed42"
        source = ROOT / "experiments" / experiment
        record = ROOT / "logs/experiments" / experiment
        archive = ROOT / "Modelos salvos" / f"transformer_mixed25_{mode}"
        checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
        if (checkpoint.get("feature_mode") != mode or checkpoint.get("num_classes") != 25 or
                checkpoint.get("model_type") != "transformer"):
            raise ValueError(f"Checkpoint incompatível: {source / 'best.pt'}")
        copy(source / "best.pt", archive / "best.pt")
        for config in config_chain:
            copy(ROOT / "configs" / config, archive / "config" / config)
        for filename in ("manifest.json", "splits.json", "label_map.json", "dataset_meta.json"):
            copy(ROOT / "data/processed/mixed_25" / filename, archive / "data_split" / filename)
        for filename in ("run_record.json", "history.json", "train.log"):
            copy(record / filename, archive / "records" / filename)
        copy(comparison_path, archive / "records/comparison.json")
        val = comparison["validation"]["face16"][name]
        test = comparison["test_exploratory"]["face16"][name]
        readme = (
            f"# Transformer mixed25 {mode}\n\n"
            "Treinado do zero com 25 classes: 15 do MINDS-Libras e dez gravadas pelo grupo. "
            "Mesmo split (500 treino, 85 validação, 85 teste), seed 42, 48 quadros, "
            "arquitetura de 122.123 parâmetros e parâmetros de treinamento do modelo de 40 pontos. "
            f"Modo `{mode}`: somente 16 pontos faciais em sobrancelhas, olhos e boca; "
            "os demais 24 pontos ficam zerados antes e depois da normalização. "
            "O modelo `pose_face16` também zera os pontos detalhados das mãos.\n\n"
            f"Validação: {val['correct']}/85 ({val['accuracy']:.1%}), F1 macro {val['f1_macro']:.3f}. "
            f"Teste exploratório já consultado anteriormente: {test['correct']}/85 "
            f"({test['accuracy']:.1%}), F1 macro {test['f1_macro']:.3f}.\n\n"
            "Conteúdo: `best.pt`; `config/` com toda a cadeia YAML; `data_split/` com "
            "manifesto, splits, mapa de classes e metadados; `records/` com histórico, "
            "registro efetivo, log e comparação. Os vídeos e `.npz` originais não são duplicados. "
            "O checkpoint foi escolhido pela menor perda de validação; o teste não orientou o treino. "
            "Veja `docs/PLANO_ABLACAO_FACE16_MIXED25_2026-11-04.md` para comandos e limites.\n"
        )
        (archive / "README.md").write_text(readme, encoding="utf-8")
        checksum = digest(archive / "best.pt")
        (archive / "checksums.sha256").write_text(f"{checksum}  best.pt\n", encoding="ascii")
        paths[name] = archive / "best.pt"

    new_spec = dict(baseline_spec)
    new_spec.update({
        "scope": "experimental 16 face points; same 25-class vocabulary",
        "selection_reason": "fixed 40-point fusion rule for controlled ablation; no new tuning",
        "pose_checkpoint": paths["pose_face"].relative_to(ROOT).as_posix(),
        "full_checkpoint": paths["full"].relative_to(ROOT).as_posix(),
        "pose_sha256": digest(paths["pose_face"]),
        "full_sha256": digest(paths["full"]),
        "label_map": (paths["full"].parent / "data_split/label_map.json").relative_to(ROOT).as_posix(),
        "label_map_sha256": digest(paths["full"].parent / "data_split/label_map.json"),
        "validation_predictions_path": "experiments/fusion_mixed25_face16/val_pose_face_predictions.csv",
        "validation_predictions_sha256": digest(ROOT / "experiments/fusion_mixed25_face16/val_pose_face_predictions.csv"),
        "face_indices": list(FACE16_MEDIAPIPE_INDICES),
        "limitations": "Experimental ablation; temperatures and weights inherited from the 40-point fusion. "
                       "The 85 test videos were consulted in prior experiments. No unseen signer validation.",
    })
    new_spec.pop("validation", None)
    destination = ROOT / "configs/fusion25_face16_fixed_rule.json"
    destination.write_text(json.dumps(new_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modelos salvos em {paths['pose_face'].parent} e {paths['full'].parent}")
    print(f"Manifesto experimental: {destination}")


if __name__ == "__main__":
    main()
