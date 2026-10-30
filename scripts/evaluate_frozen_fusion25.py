#!/usr/bin/env python3
"""Apply the already-frozen fusion rule without fitting any test parameters."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_fusion25 import apply_temperature, candidate_fusion


def evaluate(spec_path: Path, csv_path: Path) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    labels = json.loads(Path(spec["label_map"]).read_text(encoding="utf-8"))
    names = [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["top_k"] == "5"]
    if len(rows) != 85 or len({row["path"] for row in rows}) != 85:
        raise ValueError("Esperados 85 vídeos exclusivos do split de teste")
    if any("Sinalizador11" in row["path"] or "Sinalizador06" in row["path"] for row in rows):
        raise ValueError("CSV contém amostras de validação")
    truth = np.asarray([labels[row["true"]] for row in rows])
    new = np.asarray([row["path"].startswith("../novas_") for row in rows])
    def matrix(key):
        return np.asarray([[json.loads(row[key])[name] for name in names] for row in rows])
    pose = apply_temperature(matrix("pose_probabilities"), spec["temperatures"]["pose_face"])
    full = apply_temperature(matrix("full_probabilities"), spec["temperatures"]["full"])
    if spec["selected_mode"] == "fusion":
        rule = spec["fusion"]
        fused = candidate_fusion(pose, full, rule["first"], rule["top_k"], rule["pose_weight"])
    else:
        fused = full
    def score(prob):
        pred = prob.argmax(axis=1)
        return {"correct": int((pred == truth).sum()),
                "accuracy": float(np.mean(pred == truth)),
                "f1_macro": float(f1_score(truth, pred, labels=np.arange(25), average="macro", zero_division=0)),
                "new_correct": int((pred[new] == truth[new]).sum())}
    return {"split": "previously inspected test (exploratory)", "samples": len(rows),
            "selected_mode": spec["selected_mode"], "full": score(full),
            "frozen_rule": score(fused)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/fusion25_frozen.json"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("experiments/fusion_mixed25/test_pose_face_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/fusion25_frozen_test_exploratory.json"))
    args = parser.parse_args()
    report = evaluate(args.manifest, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
