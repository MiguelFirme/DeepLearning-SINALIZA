#!/usr/bin/env python3
"""One-time exploratory test of frozen 26-class rest gate on held-out signer."""
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
from ml.features.normalization import LandmarkNormalizer, NormalizationConfig
from ml.features.sequence import SequenceProcessor, SequenceConfig
from scripts.calibrate_fusion25 import apply_temperature, candidate_fusion
from scripts.evaluate_idle_edge_pilot import load_model


def frozen_predictions(spec_path: Path, predictions_csv: Path) -> dict[str, str]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    labels = json.loads(Path(spec["label_map"]).read_text(encoding="utf-8"))
    names = [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]
    result = {}
    with predictions_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["top_k"] != "5":
                continue
            pose = np.array([[json.loads(row["pose_probabilities"])[name] for name in names]])
            full = np.array([[json.loads(row["full_probabilities"])[name] for name in names]])
            pose = apply_temperature(pose, spec["temperatures"]["pose_face"])
            full = apply_temperature(full, spec["temperatures"]["full"])
            if spec["selected_mode"] == "fusion":
                rule = spec["fusion"]
                score = candidate_fusion(pose, full, rule["first"], rule["top_k"], rule["pose_weight"])
            else:
                score = full
            result[row["path"]] = names[int(score.argmax())]
    if len(result) != 85:
        raise ValueError("Esperadas 85 predições positivas de teste")
    return result


def evaluate(checkpoint: Path, data_dir: Path, processed_dir: Path, candidates_csv: Path,
             fusion_spec: Path, fusion_predictions: Path, output_dir: Path, device: str) -> dict:
    label_map = json.loads((processed_dir / "label_map.json").read_text(encoding="utf-8"))
    splits = json.loads((processed_dir / "splits.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if len(label_map) != 25:
        raise ValueError("Teste requer as 25 classes originais")
    frozen = frozen_predictions(fusion_spec, fusion_predictions)
    model = load_model(checkpoint, device)
    dataset = LibrasDataset(data_dir, label_map=label_map, split_indices=splits["test"],
                            feature_mode="full", target_frames=48)
    positive = []
    with torch.inference_mode():
        for sample in dataset:
            logits = model(sample["sequence"].unsqueeze(0).to(device),
                           sample["mask"].unsqueeze(0).to(device))
            p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            true = sample["sign_name"]
            fusion_prediction = frozen[sample["path"]]
            positive.append({"path": sample["path"], "true": true,
                             "fusion_prediction": fusion_prediction,
                             "gate_reject": int(p.argmax()) == 25,
                             "gate_reject_probability": float(p[25])})
        with candidates_csv.open(encoding="utf-8", newline="") as handle:
            candidates = list(csv.DictReader(handle))
        normalizer = LandmarkNormalizer(346, NormalizationConfig())
        seq_processor = SequenceProcessor(SequenceConfig(target_frames=48))
        negative = []
        test_set = set(splits["test"])
        candidate_lengths = set()
        for row in candidates:
            source_index = int(row["manifest_index"])
            if row["split"] != "test" or source_index not in test_set:
                raise ValueError("Candidato não pertence ao split de teste")
            sample = manifest[source_index]
            with np.load(data_dir / sample["path"]) as data:
                start, end = int(row["start_frame"]), int(row["end_frame_exclusive"])
                landmarks = data["landmarks"][start:end]
                mask = data["mask"][start:end]
            if len(landmarks) not in {32, 48}:
                raise ValueError("Janela de teste precisa ter 32 ou 48 quadros")
            candidate_lengths.add(len(landmarks))
            seq, seq_mask = seq_processor.process(normalizer.normalize_sequence(landmarks, mask))
            logits = model(torch.from_numpy(seq).unsqueeze(0).to(device),
                           torch.from_numpy(seq_mask).unsqueeze(0).to(device))
            p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            negative.append({"source_path": sample["path"], "signer": sample["signer"],
                             "gate_reject": int(p.argmax()) == 25,
                             "gate_reject_probability": float(p[25])})
    report = {"status": "exploratory test; old positive test previously inspected; weak negatives",
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "candidate_csv_sha256": hashlib.sha256(candidates_csv.read_bytes()).hexdigest(),
              "positive_samples": len(positive), "negative_candidates": len(negative),
              "negative_window_lengths": sorted(candidate_lengths),
              "fusion_positive_correct": sum(r["fusion_prediction"] == r["true"] for r in positive),
              "cascade_positive_correct": sum(r["fusion_prediction"] == r["true"] and not r["gate_reject"] for r in positive),
              "gate_false_rejections": sum(r["gate_reject"] for r in positive),
              "gate_negative_rejections": sum(r["gate_reject"] for r in negative),
              "negative_signers": sorted(set(r["signer"] for r in negative)),
              "limitations": "Same-video boundary snippets are not independent negatives; 32-frame windows are resampled to 48."}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "exploratory_test_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, rows in (("positive_test_predictions.csv", positive),
                       ("negative_test_predictions.csv", negative)):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("experiments/transformer_mixed25_edge32_rejection_pilot_seed42/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/landmarks/mixed_25"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/mixed_25"))
    parser.add_argument("--candidates", type=Path, default=Path("reports/idle_edges25/test_candidates_frozen.csv"))
    parser.add_argument("--fusion-spec", type=Path, default=Path("configs/fusion25_frozen.json"))
    parser.add_argument("--fusion-predictions", type=Path,
                        default=Path("experiments/fusion_mixed25/test_pose_face_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/idle_edges25/pilot_evaluation"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.data_dir, args.processed_dir,
                              args.candidates, args.fusion_spec, args.fusion_predictions,
                              args.output_dir, args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
