#!/usr/bin/env python3
"""Evaluate frozen 25-class pose_face40 + full_face16 fusion."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.calibrate_fusion25 import apply_temperature, candidate_fusion
from scripts.evaluate_hybrid_face16_full40 import load_rows, matrix, measures, sha256


ROOT = Path(__file__).resolve().parents[1]
BASE_SPEC = ROOT / "configs/fusion25_frozen.json"
FACE16_SPEC = ROOT / "configs/fusion25_face16_fixed_rule.json"
OUTPUT = ROOT / "experiments/fusion_mixed25_hybrid_face40_full16"
REPORT = ROOT / "reports/fusion_mixed25_hybrid_face40_full16/comparison.json"
MANIFEST = ROOT / "configs/fusion25_hybrid_face40_full16.json"


def evaluate_split(split: str, spec: dict, names: list[str], label_map: dict[str, int]) -> dict:
    old = load_rows(ROOT / f"experiments/fusion_mixed25/{split}_pose_face_predictions.csv")
    reduced = load_rows(ROOT / f"experiments/fusion_mixed25_face16/{split}_pose_face_predictions.csv")
    previous_hybrid = load_rows(ROOT / f"experiments/fusion_mixed25_hybrid/{split}_pose_face_predictions.csv")
    identity = [(row["path"], row["true"]) for row in old]
    if any([(row["path"], row["true"]) for row in rows] != identity
           for rows in (reduced, previous_hybrid)):
        raise ValueError(f"Different videos or labels in {split}")
    truth = np.asarray([label_map[row["true"]] for row in old])
    new_mask = np.asarray([row["path"].startswith("../novas_") for row in old])
    if len(names) != 25 or int(new_mask.sum()) != 10:
        raise ValueError(f"25 classes and 10 new-class examples required in {split}")

    # The baseline CSV contains raw probabilities; the face16 CSV was already
    # temperature-adjusted when it was generated. Do not calibrate it twice.
    pose40 = apply_temperature(matrix(old, "pose_probabilities", names), spec["temperatures"]["pose_face"])
    full40 = apply_temperature(matrix(old, "full_probabilities", names), spec["temperatures"]["full"])
    full16 = matrix(reduced, "full_probabilities", names)
    pose16 = matrix(reduced, "pose_probabilities", names)
    for scores in (pose40, full40, pose16, full16):
        if not np.isfinite(scores).all() or not np.allclose(scores.sum(axis=1), 1, atol=1e-5):
            raise ValueError(f"Invalid probability vectors in {split}")
    rule = spec["fusion"]
    fuse = lambda pose, full: candidate_fusion(pose, full, rule["first"], rule["top_k"], rule["pose_weight"])
    baseline = fuse(pose40, full40)
    hybrid_old = fuse(pose16, full40)
    hybrid_new = fuse(pose40, full16)
    previous_prediction = [row["fused_prediction"] for row in previous_hybrid]
    if previous_prediction != [names[int(row.argmax())] for row in hybrid_old]:
        raise ValueError(f"Previous hybrid predictions changed in {split}")
    result = {
        "samples": len(old), "new_class_samples": int(new_mask.sum()),
        "baseline_40_40": measures(baseline, truth, new_mask, names),
        "hybrid_16_40": measures(hybrid_old, truth, new_mask, names),
        "hybrid_40_16": measures(hybrid_new, truth, new_mask, names),
    }
    new_correct = hybrid_new.argmax(axis=1) == truth
    for name, other in (("baseline_40_40", baseline), ("hybrid_16_40", hybrid_old)):
        other_correct = other.argmax(axis=1) == truth
        result[f"paired_vs_{name}"] = {
            "corrected": int((new_correct & ~other_correct).sum()),
            "regressed": int((~new_correct & other_correct).sum()),
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / f"{split}_pose_face_predictions.csv"
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ["top_k", "path", "true", "pose_prediction", "full_prediction", "candidates",
                      "truth_in_candidates", "fused_prediction", "fused_correct", "fused_score",
                      "pose_probabilities", "full_probabilities"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(old):
            candidates = np.argsort(-pose40[i], kind="stable")[:rule["top_k"]]
            selected = int(hybrid_new[i].argmax())
            writer.writerow({
                "top_k": rule["top_k"], "path": row["path"], "true": row["true"],
                "pose_prediction": names[int(pose40[i].argmax())],
                "full_prediction": names[int(full16[i].argmax())],
                "candidates": json.dumps([names[j] for j in candidates], ensure_ascii=False),
                "truth_in_candidates": bool(truth[i] in candidates),
                "fused_prediction": names[selected], "fused_correct": bool(new_correct[i]),
                "fused_score": float(hybrid_new[i, selected]),
                "pose_probabilities": json.dumps(dict(zip(names, map(float, pose40[i]))), ensure_ascii=False),
                "full_probabilities": json.dumps(dict(zip(names, map(float, full16[i]))), ensure_ascii=False),
            })
    result["predictions_path"] = str(output_path.relative_to(ROOT)).replace("\\", "/")
    return result


def main() -> None:
    spec = json.loads(BASE_SPEC.read_text(encoding="utf-8"))
    face16_spec = json.loads(FACE16_SPEC.read_text(encoding="utf-8"))
    if spec["selected_mode"] != "fusion" or spec["fusion"] != {"first": "pose_face", "top_k": 5, "pose_weight": 0.22}:
        raise ValueError("Original fusion protocol changed")
    if face16_spec["fusion"] != spec["fusion"] or face16_spec["temperatures"] != spec["temperatures"]:
        raise ValueError("Face16 data was evaluated with different calibration")
    for frozen in (spec, face16_spec):
        for key, hash_key in (("pose_checkpoint", "pose_sha256"),
                              ("full_checkpoint", "full_sha256"),
                              ("label_map", "label_map_sha256"),
                              ("split_path", "split_sha256"),
                              ("validation_predictions_path", "validation_predictions_sha256")):
            if sha256(ROOT / frozen[key]) != frozen[hash_key]:
                raise ValueError(f"Frozen source changed: {frozen[key]}")
    label_map = json.loads((ROOT / spec["label_map"]).read_text(encoding="utf-8"))
    names = [name for name, _ in sorted(label_map.items(), key=lambda pair: pair[1])]
    report = {
        "protocol": "frozen pose_face40 top5 + full_face16, same 25 classes, no retuning",
        "fusion_rule": spec["fusion"], "temperatures": spec["temperatures"],
        "test_caveat": "Test videos were consulted in previous experiments; exploratory only.",
        "validation": evaluate_split("val", spec, names, label_map),
        "test_exploratory": evaluate_split("test", spec, names, label_map),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hybrid_spec = spec.copy()
    hybrid_spec.update({
        "scope": "experimental pose_face40 + full_face16; same 25-class vocabulary",
        "full_checkpoint": face16_spec["full_checkpoint"],
        "full_sha256": sha256(ROOT / face16_spec["full_checkpoint"]),
        "validation_predictions_path": "experiments/fusion_mixed25_hybrid_face40_full16/val_pose_face_predictions.csv",
        "validation_predictions_sha256": sha256(OUTPUT / "val_pose_face_predictions.csv"),
        "selection_reason": "fixed 40-point fusion rule for inverted hybrid comparison; no new tuning",
        "limitations": "Experimental inverted hybrid. Inherited temperatures/weight; test videos previously consulted. No unseen-signer validation.",
        "face_indices": face16_spec["face_indices"],
        "validation": {"hybrid_40_16": report["validation"]["hybrid_40_16"]},
    })
    MANIFEST.write_text(json.dumps(hybrid_spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split in ("validation", "test_exploratory"):
        result = report[split]
        print(split, "/".join(str(result[name]["correct"]) for name in
              ("baseline_40_40", "hybrid_16_40", "hybrid_40_16")), "of", result["samples"])
    print(f"Report: {REPORT.relative_to(ROOT)}; manifest: {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
