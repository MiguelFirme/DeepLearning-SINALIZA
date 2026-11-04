#!/usr/bin/env python3
"""Evaluate the frozen 25-class pose_face16 + full40 hybrid on matching videos."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from scripts.calibrate_fusion25 import apply_temperature, candidate_fusion


ROOT = Path(__file__).resolve().parents[1]
BASE_SPEC = ROOT / "configs/fusion25_frozen.json"
FACE16_SPEC = ROOT / "configs/fusion25_face16_fixed_rule.json"
OUTPUT = ROOT / "experiments/fusion_mixed25_hybrid"
REPORT = ROOT / "reports/fusion_mixed25_hybrid/comparison.json"
MANIFEST = ROOT / "configs/fusion25_hybrid_face16_full40.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["top_k"] == "5"]
    if len(rows) != 85 or len({row["path"] for row in rows}) != 85:
        raise ValueError(f"Expected 85 unique videos in {path}")
    return sorted(rows, key=lambda row: row["path"])


def matrix(rows: list[dict], key: str, names: list[str]) -> np.ndarray:
    return np.asarray([[json.loads(row[key])[name] for name in names] for row in rows], dtype=float)


def measures(scores: np.ndarray, truth: np.ndarray, new_mask: np.ndarray, names: list[str]) -> dict:
    pred = scores.argmax(axis=1)
    correct = pred == truth
    return {
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "f1_macro": float(f1_score(truth, pred, labels=np.arange(len(names)), average="macro", zero_division=0)),
        "new_class_correct": int((correct & new_mask).sum()),
        "per_class_correct": {name: int((correct & (truth == i)).sum()) for i, name in enumerate(names)},
        "per_class_total": {name: int((truth == i).sum()) for i, name in enumerate(names)},
    }


def evaluate_split(split: str, spec: dict, names: list[str], label_map: dict[str, int]) -> dict:
    old = load_rows(ROOT / f"experiments/fusion_mixed25/{split}_pose_face_predictions.csv")
    reduced = load_rows(ROOT / f"experiments/fusion_mixed25_face16/{split}_pose_face_predictions.csv")
    if [(r["path"], r["true"]) for r in old] != [(r["path"], r["true"]) for r in reduced]:
        raise ValueError(f"Different videos or labels in {split}")
    truth = np.asarray([label_map[r["true"]] for r in old])
    new_mask = np.asarray([r["path"].startswith("../novas_") for r in old])
    if len(names) != 25 or new_mask.sum() != 10:
        raise ValueError(f"25 classes and 10 new-class examples required in {split}")

    # The face16 CSV is already temperature-adjusted by evaluate_two_stage_fusion.
    pose16 = matrix(reduced, "pose_probabilities", names)
    full16 = matrix(reduced, "full_probabilities", names)
    pose40 = apply_temperature(matrix(old, "pose_probabilities", names), spec["temperatures"]["pose_face"])
    full40 = apply_temperature(matrix(old, "full_probabilities", names), spec["temperatures"]["full"])
    for scores in (pose16, full16, pose40, full40):
        if not np.isfinite(scores).all() or not np.allclose(scores.sum(axis=1), 1, atol=1e-5):
            raise ValueError(f"Invalid probability vectors in {split}")
    rule = spec["fusion"]
    fuse = lambda pose, full: candidate_fusion(pose, full, rule["first"], rule["top_k"], rule["pose_weight"])
    baseline = fuse(pose40, full40)
    face16 = fuse(pose16, full16)
    hybrid = fuse(pose16, full40)
    measures_by_arch = {
        "baseline_40_40": measures(baseline, truth, new_mask, names),
        "face16_16": measures(face16, truth, new_mask, names),
        "hybrid_16_40": measures(hybrid, truth, new_mask, names),
    }
    hybrid_correct = hybrid.argmax(axis=1) == truth
    old_correct = baseline.argmax(axis=1) == truth
    measures_by_arch["paired_vs_baseline"] = {
        "corrected": int((hybrid_correct & ~old_correct).sum()),
        "regressed": int((~hybrid_correct & old_correct).sum()),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / f"{split}_pose_face_predictions.csv"
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ["top_k", "path", "true", "pose_prediction", "full_prediction", "candidates",
                      "truth_in_candidates", "fused_prediction", "fused_correct", "fused_score",
                      "pose_probabilities", "full_probabilities"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for i, original in enumerate(old):
            candidates = np.argsort(-pose16[i], kind="stable")[:rule["top_k"]]
            selected = int(hybrid[i].argmax())
            writer.writerow({
                "top_k": rule["top_k"], "path": original["path"], "true": original["true"],
                "pose_prediction": names[int(pose16[i].argmax())],
                "full_prediction": names[int(full40[i].argmax())],
                "candidates": json.dumps([names[j] for j in candidates], ensure_ascii=False),
                "truth_in_candidates": bool(truth[i] in candidates),
                "fused_prediction": names[selected], "fused_correct": bool(hybrid_correct[i]),
                "fused_score": float(hybrid[i, selected]),
                "pose_probabilities": json.dumps(dict(zip(names, map(float, pose16[i]))), ensure_ascii=False),
                "full_probabilities": json.dumps(dict(zip(names, map(float, full40[i]))), ensure_ascii=False),
            })
    return {"samples": len(old), "new_class_samples": int(new_mask.sum()),
            **measures_by_arch, "predictions_path": str(output_path.relative_to(ROOT)).replace("\\", "/")}


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
        "protocol": "frozen pose_face16 top5 + full40, same 25 classes, no retuning",
        "fusion_rule": spec["fusion"], "temperatures": spec["temperatures"],
        "test_caveat": "Test videos were consulted in previous experiments; exploratory only.",
        "validation": evaluate_split("val", spec, names, label_map),
        "test_exploratory": evaluate_split("test", spec, names, label_map),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hybrid_spec = spec.copy()
    hybrid_spec.update({
        "scope": "experimental pose_face16 + full40; same 25-class vocabulary",
        "pose_checkpoint": face16_spec["pose_checkpoint"],
        "pose_sha256": sha256(ROOT / face16_spec["pose_checkpoint"]),
        "validation_predictions_path": "experiments/fusion_mixed25_hybrid/val_pose_face_predictions.csv",
        "validation_predictions_sha256": sha256(OUTPUT / "val_pose_face_predictions.csv"),
        "selection_reason": "fixed 40-point fusion rule for hybrid comparison; no new tuning",
        "limitations": "Experimental hybrid. Inherited temperatures/weight; test videos previously consulted. No unseen-signer validation.",
        "face_indices": face16_spec["face_indices"],
        "validation": {"hybrid_16_40": report["validation"]["hybrid_16_40"]},
    })
    MANIFEST.write_text(json.dumps(hybrid_spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split in ("validation", "test_exploratory"):
        result = report[split]
        print(split, "/".join(str(result[name]["correct"]) for name in
              ("baseline_40_40", "face16_16", "hybrid_16_40")), "of", result["samples"])
    print(f"Report: {REPORT.relative_to(ROOT)}; manifest: {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
