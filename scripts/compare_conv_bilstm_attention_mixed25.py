#!/usr/bin/env python3
"""Compare the new 25-class ConvBiLSTM pair with the frozen Transformer pair."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.calibrate_fusion25 import apply_temperature, candidate_fusion
from scripts.evaluate_hybrid_face16_full40 import load_rows, matrix, measures


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/fusion_mixed25_conv_bilstm_attention/comparison.json"


def compare_split(split: str, spec: dict, names: list[str], label_map: dict[str, int]) -> dict:
    old = load_rows(ROOT / f"experiments/fusion_mixed25/{split}_pose_face_predictions.csv")
    new = load_rows(ROOT / f"experiments/fusion_mixed25_conv_bilstm_attention/{split}_pose_face_predictions.csv")
    if [(row["path"], row["true"]) for row in old] != [(row["path"], row["true"]) for row in new]:
        raise ValueError(f"Videos or labels differ for {split}")
    truth = np.asarray([label_map[row["true"]] for row in old])
    new_mask = np.asarray([row["path"].startswith("../novas_") for row in old])
    if len(names) != 25 or new_mask.sum() != 10:
        raise ValueError(f"Expected 25 classes and 10 new-class examples in {split}")
    rule = spec["fusion"]
    old_pose = apply_temperature(matrix(old, "pose_probabilities", names), spec["temperatures"]["pose_face"])
    old_full = apply_temperature(matrix(old, "full_probabilities", names), spec["temperatures"]["full"])
    new_pose = matrix(new, "pose_probabilities", names)
    new_full = matrix(new, "full_probabilities", names)
    old_fusion = candidate_fusion(old_pose, old_full, rule["first"], rule["top_k"], rule["pose_weight"])
    new_fusion = candidate_fusion(new_pose, new_full, rule["first"], rule["top_k"], rule["pose_weight"])
    old_correct = old_fusion.argmax(axis=1) == truth
    new_correct = new_fusion.argmax(axis=1) == truth
    return {
        "samples": len(truth), "new_class_samples": int(new_mask.sum()),
        "transformer": {
            "pose_face": measures(old_pose, truth, new_mask, names),
            "full": measures(old_full, truth, new_mask, names),
            "fusion": measures(old_fusion, truth, new_mask, names),
        },
        "conv_bilstm_attention": {
            "pose_face": measures(new_pose, truth, new_mask, names),
            "full": measures(new_full, truth, new_mask, names),
            "fusion": measures(new_fusion, truth, new_mask, names),
        },
        "fusion_paired": {
            "corrected_by_new": int((new_correct & ~old_correct).sum()),
            "regressed_by_new": int((~new_correct & old_correct).sum()),
        },
    }


def main() -> None:
    spec = json.loads((ROOT / "configs/fusion25_frozen.json").read_text(encoding="utf-8"))
    if spec["selected_mode"] != "fusion" or spec["fusion"] != {"first": "pose_face", "top_k": 5, "pose_weight": 0.22}:
        raise ValueError("Frozen Transformer rule changed")
    labels = json.loads((ROOT / spec["label_map"]).read_text(encoding="utf-8"))
    names = [name for name, _ in sorted(labels.items(), key=lambda pair: pair[1])]
    report = {
        "protocol": "same 25 classes/48 frames/splits; original Transformer fusion vs Conv1D+BiLSTM+attention",
        "rule": spec["fusion"],
        "temperatures": {"transformer": spec["temperatures"],
                         "conv_bilstm_attention": {"pose_face": 1.0, "full": 1.0}},
        "test_caveat": "Test videos were previously consulted; result is exploratory.",
        "validation": compare_split("val", spec, names, labels),
        "test_exploratory": compare_split("test", spec, names, labels),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split in ("validation", "test_exploratory"):
        result = report[split]
        print(split, "transformer", result["transformer"]["fusion"]["correct"],
              "new", result["conv_bilstm_attention"]["fusion"]["correct"],
              "paired", result["fusion_paired"])
    print(f"Report: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
