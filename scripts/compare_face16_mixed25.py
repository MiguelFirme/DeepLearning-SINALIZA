#!/usr/bin/env python3
"""Compare the frozen 40-face-point pair with the 16-point pair, path by path."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.calibrate_fusion25 import apply_temperature, candidate_fusion


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["top_k"] == "5"]
    if len(rows) != 85 or len({row["path"] for row in rows}) != 85:
        raise ValueError(f"Esperados 85 vídeos únicos em {path}")
    return sorted(rows, key=lambda row: row["path"])


def _matrix(rows: list[dict], key: str, names: list[str]) -> np.ndarray:
    return np.asarray([[json.loads(row[key])[name] for name in names] for row in rows])


def _score(probabilities: np.ndarray, truth: np.ndarray, new_mask: np.ndarray,
           names: list[str]) -> dict:
    prediction = probabilities.argmax(axis=1)
    matrix = confusion_matrix(truth, prediction, labels=np.arange(len(names)))
    return {
        "correct": int((prediction == truth).sum()),
        "accuracy": float(np.mean(prediction == truth)),
        "f1_macro": float(f1_score(truth, prediction, labels=np.arange(len(names)),
                                   average="macro", zero_division=0)),
        "new_class_correct": int(((prediction == truth) & new_mask).sum()),
        "per_class_correct": {name: int(matrix[index, index]) for index, name in enumerate(names)},
        "per_class_total": {name: int(matrix[index].sum()) for index, name in enumerate(names)},
    }


def compare_split(baseline_path: Path, face16_path: Path, spec: dict,
                  label_map: dict[str, int]) -> dict:
    baseline = _rows(baseline_path)
    reduced = _rows(face16_path)
    if [(row["path"], row["true"]) for row in baseline] != [
        (row["path"], row["true"]) for row in reduced
    ]:
        raise ValueError("Vídeos ou rótulos diferem entre 40 e 16 pontos")
    names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    truth = np.asarray([label_map[row["true"]] for row in baseline])
    new_mask = np.asarray([row["path"].startswith("../novas_") for row in baseline])
    rule = spec["fusion"]
    result = {"samples": len(truth), "new_class_samples": int(new_mask.sum()),
              "baseline_40": {}, "face16": {}, "difference_face16_minus_40": {}}
    for key, rows in (("baseline_40", baseline), ("face16", reduced)):
        pose = _matrix(rows, "pose_probabilities", names)
        full = _matrix(rows, "full_probabilities", names)
        if key == "baseline_40":
            pose = apply_temperature(pose, spec["temperatures"]["pose_face"])
            full = apply_temperature(full, spec["temperatures"]["full"])
        fusion = candidate_fusion(pose, full, rule["first"], rule["top_k"],
                                  rule["pose_weight"])
        for mode, scores in (("pose_face", pose), ("full", full), ("fusion", fusion)):
            result[key][mode] = _score(scores, truth, new_mask, names)
    for mode in ("pose_face", "full", "fusion"):
        old = result["baseline_40"][mode]
        new = result["face16"][mode]
        result["difference_face16_minus_40"][mode] = {
            "correct": new["correct"] - old["correct"],
            "accuracy_points": 100 * (new["accuracy"] - old["accuracy"]),
            "f1_macro": new["f1_macro"] - old["f1_macro"],
            "new_class_correct": new["new_class_correct"] - old["new_class_correct"],
            "per_class_correct": {name: new["per_class_correct"][name] - old["per_class_correct"][name]
                                  for name in names},
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, default=Path("configs/fusion25_frozen.json"))
    parser.add_argument("--baseline-val", type=Path,
                        default=Path("experiments/fusion_mixed25/val_pose_face_predictions.csv"))
    parser.add_argument("--baseline-test", type=Path,
                        default=Path("experiments/fusion_mixed25/test_pose_face_predictions.csv"))
    parser.add_argument("--face16-val", type=Path,
                        default=Path("experiments/fusion_mixed25_face16/val_pose_face_predictions.csv"))
    parser.add_argument("--face16-test", type=Path,
                        default=Path("experiments/fusion_mixed25_face16/test_pose_face_predictions.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/fusion_mixed25_face16/comparison.json"))
    args = parser.parse_args()
    spec = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    if spec["selected_mode"] != "fusion" or spec["fusion"] != {
        "first": "pose_face", "top_k": 5, "pose_weight": 0.22
    }:
        raise ValueError("Regra baseline diferente do protocolo pré-fixado")
    label_map = json.loads(Path(spec["label_map"]).read_text(encoding="utf-8"))
    report = {
        "protocol": "mesmo split, seed e arquitetura; 40 versus 16 pontos faciais",
        "fusion_rule": spec["fusion"], "baseline_temperatures": spec["temperatures"],
        "test_caveat": "teste já consultado em experimentos anteriores; resultado exploratório",
        "validation": compare_split(args.baseline_val, args.face16_val, spec, label_map),
        "test_exploratory": compare_split(args.baseline_test, args.face16_test, spec, label_map),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for split in ("validation", "test_exploratory"):
        print(split)
        for mode in ("pose_face", "full", "fusion"):
            old = report[split]["baseline_40"][mode]
            new = report[split]["face16"][mode]
            print(f"  {mode}: 40={old['correct']}/85, 16={new['correct']}/85, "
                  f"delta={new['correct'] - old['correct']:+d}")
    print(f"Relatório: {args.output}")


if __name__ == "__main__":
    main()
