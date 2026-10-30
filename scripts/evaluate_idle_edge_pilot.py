#!/usr/bin/env python3
"""Compare 26-class weak-label pilot to frozen 25-class full model on validation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.data.dataset import LibrasDataset
from ml.models import create_model
from scripts.calibrate_fusion25 import apply_temperature, candidate_fusion


def load_model(path: Path, device: str):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    state = artifact["model_state_dict"]
    input_dim = int(state["input_norm.weight"].shape[0])
    model = create_model(artifact["model_type"], input_dim=input_dim,
                         num_classes=artifact["num_classes"], **artifact["model_kwargs"])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def evaluate(pilot_checkpoint: Path, baseline_checkpoint: Path,
             data_dir: Path, processed_dir: Path, output_dir: Path, device: str) -> dict:
    label_map = json.loads((processed_dir / "label_map.json").read_text(encoding="utf-8"))
    splits = json.loads((processed_dir / "splits.json").read_text(encoding="utf-8"))
    dataset = LibrasDataset(data_dir, label_map=label_map, split_indices=splits["val"],
                            feature_mode="full", target_frames=48)
    model26 = load_model(pilot_checkpoint, device)
    model25 = load_model(baseline_checkpoint, device)
    names = {v: k for k, v in label_map.items()}
    rows = []
    with torch.inference_mode():
        for sample in dataset:
            sequence = sample["sequence"].unsqueeze(0).to(device)
            mask = sample["mask"].unsqueeze(0).to(device)
            p26 = torch.softmax(model26(sequence, mask), dim=-1).cpu().numpy()[0]
            p25 = torch.softmax(model25(sequence, mask), dim=-1).cpu().numpy()[0]
            true = int(sample["label"])
            predicted26, predicted25 = int(p26.argmax()), int(p25.argmax())
            rows.append({"path": sample["path"], "true": names[true],
                         "is_negative": true == 25,
                         "pilot_prediction": names[predicted26],
                         "pilot_confidence": float(p26[predicted26]),
                         "baseline_prediction": names[predicted25],
                         "baseline_confidence": float(p25[predicted25]),
                         "baseline_below_030": bool(p25.max() < 0.3)})
    positive = [r for r in rows if not r["is_negative"]]
    negative = [r for r in rows if r["is_negative"]]
    if len(positive) != 85 or len(negative) != 11:
        raise ValueError("Esperados 85 positivos e 11 candidatos negativos de validação")
    spec = json.loads((ROOT / "configs/fusion25_frozen.json").read_text(encoding="utf-8"))
    names25 = [name for name, index in sorted(label_map.items(), key=lambda item: item[1]) if index < 25]
    gated = {row["path"]: row for row in positive}
    fusion_correct = cascade_correct = 0
    with (ROOT / "experiments/fusion_mixed25/val_pose_face_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if row["top_k"] != "5":
                continue
            pose = np.array([[json.loads(row["pose_probabilities"])[name] for name in names25]])
            full = np.array([[json.loads(row["full_probabilities"])[name] for name in names25]])
            pose = apply_temperature(pose, spec["temperatures"]["pose_face"])
            full = apply_temperature(full, spec["temperatures"]["full"])
            rule = spec["fusion"]
            score = candidate_fusion(pose, full, rule["first"], rule["top_k"], rule["pose_weight"])
            correct = names25[int(score.argmax())] == row["true"]
            fusion_correct += int(correct)
            cascade_correct += int(correct and gated[row["path"]]["pilot_prediction"] != "__nenhum_sinal__")
    report = {
        "split": "validation only", "status": "weak-label pilot, not deployment ready",
        "positive_samples": len(positive), "negative_samples": len(negative),
        "baseline25_positive_correct": sum(r["baseline_prediction"] == r["true"] for r in positive),
        "pilot26_positive_correct": sum(r["pilot_prediction"] == r["true"] for r in positive),
        "pilot26_false_rejections": sum(r["pilot_prediction"] == "__nenhum_sinal__" for r in positive),
        "pilot26_negative_recall_count": sum(r["pilot_prediction"] == "__nenhum_sinal__" for r in negative),
        "baseline25_heuristic_reject_negative_count": sum(r["baseline_below_030"] for r in negative),
        "baseline25_heuristic_false_reject_positive_count": sum(r["baseline_below_030"] for r in positive),
        "frozen_fusion25_positive_correct": fusion_correct,
        "cascade_positive_correct": cascade_correct,
        "pilot_checkpoint_sha256": hashlib.sha256(pilot_checkpoint.read_bytes()).hexdigest(),
        "baseline_checkpoint_sha256": hashlib.sha256(baseline_checkpoint.read_bytes()).hexdigest(),
        "limitations": "Negatives are 32-frame source clips resampled to 48, all MINDS signer11 in validation; no independently recorded negatives.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path,
                        default=Path("experiments/transformer_mixed25_edge32_rejection_pilot_seed42/best.pt"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("Modelos salvos/transformer_mixed25_full/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/landmarks/mixed25_edge32_pilot"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/mixed25_edge32_pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/idle_edges25/pilot_evaluation"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.pilot, args.baseline, args.data_dir, args.processed_dir,
                              args.output_dir, args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
