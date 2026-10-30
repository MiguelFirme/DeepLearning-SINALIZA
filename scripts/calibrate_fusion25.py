#!/usr/bin/env python3
"""Ajusta temperaturas/fusão nos 85 vídeos de VALIDAÇÃO e congela a regra."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(probs: np.ndarray, truth: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(truth)), truth], 1e-12, 1.0)).mean())


def ece(probs: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == truth
    value = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        select = (confidence >= lo) & (confidence <= hi if hi == 1 else confidence < hi)
        if select.any():
            value += select.mean() * abs(confidence[select].mean() - correct[select].mean())
    return float(value)


def metrics(probs: np.ndarray, truth: np.ndarray, new_mask: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    return {"correct": int((pred == truth).sum()),
            "f1_macro": float(f1_score(truth, pred, labels=np.arange(probs.shape[1]),
                                       average="macro", zero_division=0)),
            "new_correct": int((pred[new_mask] == truth[new_mask]).sum()),
            "nll": nll(probs, truth), "ece_10bin": ece(probs, truth)}


def classification_metrics(probs: np.ndarray, truth: np.ndarray, new_mask: np.ndarray) -> dict:
    """Top-K restricted scores do not define calibrated class probabilities."""
    result = metrics(probs, truth, new_mask)
    return {key: result[key] for key in ("correct", "f1_macro", "new_correct")}


def candidate_fusion(pose: np.ndarray, full: np.ndarray, first: str,
                     top_k: int, pose_weight: float) -> np.ndarray:
    proposal = pose if first == "pose_face" else full
    top = np.argsort(-proposal, axis=1, kind="stable")[:, :top_k]
    score = pose_weight * pose + (1 - pose_weight) * full
    result = np.full_like(score, -np.inf)
    np.put_along_axis(result, top, np.take_along_axis(score, top, axis=1), axis=1)
    result = np.where(np.isfinite(result), result, 0.0)
    return result


def calibrate(predictions_csv: Path, label_map_path: Path,
              pose_checkpoint: Path, full_checkpoint: Path,
              output: Path, split_path: Path, data_manifest_path: Path) -> dict:
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    if len(label_map) != 25:
        raise ValueError("Calibração exige exatamente as 25 classes aprovadas")
    with predictions_csv.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["top_k"] == "5"]
    if len(rows) != 85 or len({r["path"] for r in rows}) != 85:
        raise ValueError("Esperados 85 vídeos únicos de validação")
    if any("Sinalizador12" in r["path"] or "Sinalizador07" in r["path"] for r in rows):
        raise ValueError("A calibração recebeu vídeo de teste")
    names = [name for name, _ in sorted(label_map.items(), key=lambda pair: pair[1])]
    truth = np.asarray([label_map[r["true"]] for r in rows], dtype=int)
    new_mask = np.asarray([r["path"].startswith("../novas_") for r in rows], dtype=bool)
    if int(new_mask.sum()) != 10:
        raise ValueError("Validação deve conter dez vídeos novos")
    def matrix(key):
        return np.asarray([[json.loads(row[key])[name] for name in names] for row in rows], dtype=float)
    raw_pose, raw_full = matrix("pose_probabilities"), matrix("full_probabilities")
    grid = np.exp(np.linspace(np.log(0.5), np.log(5.0), 151))
    pose_t = float(min(grid, key=lambda t: nll(apply_temperature(raw_pose, t), truth)))
    full_t = float(min(grid, key=lambda t: nll(apply_temperature(raw_full, t), truth)))
    pose = apply_temperature(raw_pose, pose_t)
    full = apply_temperature(raw_full, full_t)
    weight_grid = np.linspace(0, 1, 101)
    pose_weight = float(min(weight_grid, key=lambda w: nll(w * pose + (1 - w) * full, truth)))
    full_metrics = metrics(full, truth, new_mask)
    options = {}
    for first in ("pose_face", "full"):
        score = candidate_fusion(pose, full, first, 5, pose_weight)
        options[first] = classification_metrics(score, truth, new_mask)
    eligible = [(first, result) for first, result in options.items()
                if result["f1_macro"] >= full_metrics["f1_macro"] + 0.02
                and result["new_correct"] >= full_metrics["new_correct"]
                and result["correct"] >= full_metrics["correct"]]
    selected_first = max(eligible, key=lambda pair: (pair[1]["f1_macro"],
                                                       pair[1]["correct"]))[0] if eligible else None
    report = {
        "schema_version": 1,
        "scope": "15 selected MINDS classes + 10 new classes",
        "selection_split": "val", "selection_samples": 85,
        "validation_predictions_path": str(predictions_csv.as_posix()),
        "validation_predictions_sha256": sha256(predictions_csv),
        "selected_mode": "fusion" if selected_first else "single_full",
        "selection_reason": "predeclared F1 +0.02, new-class and total nondegradation rule",
        "pose_checkpoint": str(pose_checkpoint.as_posix()),
        "full_checkpoint": str(full_checkpoint.as_posix()),
        "pose_sha256": sha256(pose_checkpoint),
        "full_sha256": sha256(full_checkpoint),
        "label_map": str(label_map_path.as_posix()),
        "label_map_sha256": sha256(label_map_path),
        "split_path": str(split_path.as_posix()),
        "split_sha256": sha256(split_path),
        "data_manifest_path": str(data_manifest_path.as_posix()),
        "data_manifest_sha256": sha256(data_manifest_path),
        "target_frames": 48,
        "temperatures": {"pose_face": pose_t, "full": full_t},
        "fusion": {"first": selected_first or max(options, key=lambda k: options[k]["f1_macro"]),
                   "top_k": 5, "pose_weight": pose_weight},
        "validation": {
            "raw_pose_face": metrics(raw_pose, truth, new_mask),
            "raw_full": metrics(raw_full, truth, new_mask),
            "calibrated_pose_face": metrics(pose, truth, new_mask),
            "calibrated_full": full_metrics,
            "fusion_pose_first": options["pose_face"],
            "fusion_full_first": options["full"],
        },
        "limitations": "Temperatures and weight fitted on the same 85 validation videos. "
                       "Fused top-5 scores are not calibrated confidence probabilities. "
                       "No reviewed negatives or unseen signers for rejection calibration.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=Path("experiments/fusion_mixed25/val_pose_face_predictions.csv"))
    parser.add_argument("--label-map", type=Path,
                        default=Path("Modelos salvos/transformer_mixed25_full/data_split/label_map.json"))
    parser.add_argument("--pose-checkpoint", type=Path,
                        default=Path("Modelos salvos/transformer_mixed25_pose_face/best.pt"))
    parser.add_argument("--full-checkpoint", type=Path,
                        default=Path("Modelos salvos/transformer_mixed25_full/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("configs/fusion25_frozen.json"))
    parser.add_argument("--split", type=Path, default=Path("data/processed/mixed_25/splits.json"))
    parser.add_argument("--data-manifest", type=Path, default=Path("data/landmarks/mixed_25/manifest.json"))
    args = parser.parse_args()
    report = calibrate(args.predictions, args.label_map,
                       args.pose_checkpoint, args.full_checkpoint, args.output,
                       args.split, args.data_manifest)
    print(json.dumps({k: report[k] for k in ("selected_mode", "temperatures", "fusion", "validation")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
