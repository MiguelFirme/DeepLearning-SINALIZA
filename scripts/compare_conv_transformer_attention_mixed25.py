#!/usr/bin/env python3
"""Compare the ConvTransformer pair with the frozen Transformer and ConvBiLSTM pairs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_fusion25 import apply_temperature, candidate_fusion
from scripts.evaluate_hybrid_face16_full40 import load_rows, matrix, measures


OUTPUT = ROOT / "reports/fusion_mixed25_conv_transformer_attention/comparison.json"
SOURCES = {
    "transformer": "experiments/fusion_mixed25",
    "conv_bilstm_attention": "experiments/fusion_mixed25_conv_bilstm_attention",
    "conv_transformer_attention": "experiments/fusion_mixed25_conv_transformer_attention",
}


def compare_split(split: str, spec: dict, names: list[str], labels: dict[str, int]) -> dict:
    rows = {key: load_rows(ROOT / folder / f"{split}_pose_face_predictions.csv")
            for key, folder in SOURCES.items()}
    order = [(row["path"], row["true"]) for row in rows["transformer"]]
    if any([(row["path"], row["true"]) for row in value] != order for value in rows.values()):
        raise ValueError(f"Video order or labels differ in {split}")
    truth = np.asarray([labels[label] for _, label in order])
    new_mask = np.asarray([path.startswith("../novas_") for path, _ in order])
    if len(names) != 25 or len(truth) != 85 or new_mask.sum() != 10:
        raise ValueError("Unexpected 25-class comparison protocol")
    result = {"samples": len(truth), "new_class_samples": int(new_mask.sum())}
    correct = {}
    for key, value in rows.items():
        pose = matrix(value, "pose_probabilities", names)
        full = matrix(value, "full_probabilities", names)
        if key == "transformer":
            pose = apply_temperature(pose, spec["temperatures"]["pose_face"])
            full = apply_temperature(full, spec["temperatures"]["full"])
        rule = spec["fusion"]
        fused = candidate_fusion(pose, full, rule["first"], rule["top_k"], rule["pose_weight"])
        result[key] = {"pose_face": measures(pose, truth, new_mask, names),
                       "full": measures(full, truth, new_mask, names),
                       "fusion": measures(fused, truth, new_mask, names)}
        correct[key] = fused.argmax(axis=1) == truth
    result["paired_vs_conv_bilstm"] = {
        "corrected_by_conv_transformer": int((correct["conv_transformer_attention"] &
                                              ~correct["conv_bilstm_attention"]).sum()),
        "regressed_by_conv_transformer": int((~correct["conv_transformer_attention"] &
                                              correct["conv_bilstm_attention"]).sum()),
    }
    return result


def main() -> None:
    spec = json.loads((ROOT / "configs/fusion25_frozen.json").read_text(encoding="utf-8"))
    if spec["fusion"] != {"first": "pose_face", "top_k": 5, "pose_weight": 0.22}:
        raise ValueError("Original fusion rule changed")
    labels = json.loads((ROOT / spec["label_map"]).read_text(encoding="utf-8"))
    names = [name for name, _ in sorted(labels.items(), key=lambda pair: pair[1])]
    report = {
        "protocol": "same 25 classes, 48 frames, and 500/85/85 video splits",
        "fusion_rule": spec["fusion"],
        "temperatures": {"transformer": spec["temperatures"],
                         "conv_bilstm_attention": {"pose_face": 1.0, "full": 1.0},
                         "conv_transformer_attention": {"pose_face": 1.0, "full": 1.0}},
        "test_caveat": "Previously consulted test videos; exploratory comparison only.",
        "validation": compare_split("val", spec, names, labels),
        "test_exploratory": compare_split("test", spec, names, labels),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split in ("validation", "test_exploratory"):
        print(split, {key: report[split][key]["fusion"]["correct"] for key in SOURCES})
    print(f"Report: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
