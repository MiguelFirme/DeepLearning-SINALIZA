#!/usr/bin/env python3
"""Archive the two trained 25-class ConvBiLSTM checkpoints and comparison rule."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/fusion_mixed25_conv_bilstm_attention"
MANIFEST = ROOT / "configs/fusion25_conv_bilstm_attention_fixed_rule.json"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if digest(source) != digest(target):
        raise ValueError(f"Copy checksum mismatch: {target}")


def main() -> None:
    baseline = json.loads((ROOT / "configs/fusion25_frozen.json").read_text(encoding="utf-8"))
    label_map = json.loads((ROOT / baseline["label_map"]).read_text(encoding="utf-8"))
    reports = {split: json.loads((RESULTS / f"{split}_pose_face_metrics.json").read_text(encoding="utf-8"))
               for split in ("val", "test")}
    checkpoints = {}
    for name in ("pose_face", "full"):
        experiment = f"conv_bilstm_attention_mixed25_{name}_seed42"
        source = ROOT / "experiments" / experiment
        logs = ROOT / "logs/experiments" / experiment
        archive = ROOT / "Modelos salvos" / f"conv_bilstm_attention_mixed25_{name}"
        record = json.loads((logs / "run_record.json").read_text(encoding="utf-8"))
        if record["status"] != "completed" or record["resolved"]["skip_test"] is not True:
            raise ValueError(f"Incomplete or test-tuned run: {experiment}")
        if (record["dataset"]["splits_sha256"] != baseline["split_sha256"] or
                record["dataset"]["manifest_sha256"] != baseline["data_manifest_sha256"] or
                record["dataset"]["label_map_sha256"] != baseline["label_map_sha256"]):
            raise ValueError(f"Different dataset protocol: {experiment}")
        checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
        if (checkpoint.get("model_type") != "conv_bilstm_attention" or
                checkpoint.get("feature_mode") != name or
                checkpoint.get("num_classes") != 25 or
                checkpoint.get("label_map") not in (None, label_map) or
                checkpoint.get("target_frames") != 48):
            raise ValueError(f"Incompatible checkpoint: {experiment}")
        copy(source / "best.pt", archive / "best.pt")
        for config in ("default.yaml", "conv_bilstm_attention_mixed25_pose_face.yaml",
                       "conv_bilstm_attention_mixed25_full.yaml"):
            copy(ROOT / "configs" / config, archive / "config" / config)
        for filename in ("manifest.json", "splits.json", "label_map.json", "dataset_meta.json"):
            copy(ROOT / "data/processed/mixed_25" / filename, archive / "data_split" / filename)
        for filename in ("run_record.json", "history.json", "train.log"):
            copy(logs / filename, archive / "records" / filename)
        for split in ("val", "test"):
            copy(RESULTS / f"{split}_pose_face_metrics.json", archive / "records" / f"{split}_metrics.json")
        val = reports["val"][name]
        test = reports["test"][name]
        readme = (
            f"# Conv1D + BiLSTM + atenção: {name}\n\n"
            "Treino do zero com 25 classes (15 sinais MINDS-Libras e dez gravados pelo grupo), "
            "500/85/85 vídeos nos mesmos splits da dupla Transformer, seed 42 e 48 quadros. "
            "Entrada de 697 features derivadas dos landmarks MediaPipe; "
            f"modo `{name}` com 40 pontos faciais. "
            "Projeção 64 → Conv1D 64 canais/kernel 3 → duas BiLSTMs de 32 unidades por direção "
            "→ atenção temporal → classificador.\n\n"
            f"Validação: {val['correct']}/85, F1 macro {val['f1_macro']:.3f}. "
            f"Teste exploratório já consultado anteriormente: {test['correct']}/85, "
            f"F1 macro {test['f1_macro']:.3f}. "
            "O checkpoint foi escolhido exclusivamente pela menor `val_loss`; o teste não orientou o treino.\n\n"
            "Arquivos: `best.pt`, `checksums.sha256`, `config/`, `data_split/` e `records/`. "
            "Vídeos e `.npz` não são duplicados. "
            "Veja `docs/RELATORIO_CONV_BILSTM_ATENCAO_MIXED25_2026-11-04.md` para a comparação completa.\n"
        )
        (archive / "README.md").write_text(readme, encoding="utf-8")
        (archive / "checksums.sha256").write_text(f"{digest(archive / 'best.pt')}  best.pt\n", encoding="ascii")
        checkpoints[name] = archive / "best.pt"

    new_spec = dict(baseline)
    new_spec.update({
        "scope": "experimental Conv1D + BiLSTM + attention, same 25 classes and 40 face points",
        "selection_reason": "fixed original top-5/0.22 rule; no calibration on test",
        "pose_checkpoint": checkpoints["pose_face"].relative_to(ROOT).as_posix(),
        "full_checkpoint": checkpoints["full"].relative_to(ROOT).as_posix(),
        "pose_sha256": digest(checkpoints["pose_face"]),
        "full_sha256": digest(checkpoints["full"]),
        "temperatures": {"pose_face": 1.0, "full": 1.0},
        "validation_predictions_path": "experiments/fusion_mixed25_conv_bilstm_attention/val_pose_face_predictions.csv",
        "validation_predictions_sha256": digest(RESULTS / "val_pose_face_predictions.csv"),
        "validation": {"pose_face": reports["val"]["pose_face"],
                       "full": reports["val"]["full"],
                       "fusion": reports["val"]["fusion"]["5"]},
        "limitations": "Experimental. Fusion weight inherited from Transformer pair; temperatures 1.0. "
                       "The 85 test videos were consulted previously. No unseen-signer or live-camera validation.",
    })
    MANIFEST.write_text(json.dumps(new_spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Archived: {checkpoints['pose_face'].parent}, {checkpoints['full'].parent}")
    print(f"Manifest: {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
