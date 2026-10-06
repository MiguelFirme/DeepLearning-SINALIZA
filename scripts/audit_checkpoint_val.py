#!/usr/bin/env python3
"""Salva previsoes e matriz de confusao de um checkpoint em val ou test."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data.dataset import LibrasDataset
from ml.models import create_model
from ml.training.metrics import MetricsCalculator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.checkpoint.parent
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    label_map = json.loads((args.processed_dir / "label_map.json").read_text(encoding="utf-8"))
    splits = json.loads((args.processed_dir / "splits.json").read_text(encoding="utf-8"))
    inverse = {value: key for key, value in label_map.items()}
    dataset = LibrasDataset(args.data_dir, label_map=label_map,
                            target_frames=int(checkpoint.get("target_frames", 48)),
                            split_indices=splits[args.split],
                            feature_mode=checkpoint.get("feature_mode", "full"))
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    model_type = checkpoint.get("model_type") or checkpoint["model_class"].lower().replace("model", "")
    model = create_model(model_type, input_dim=dataset.feature_dim,
                         num_classes=len(label_map), **checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    calc = MetricsCalculator(len(label_map), inverse)
    rows = []
    total_loss = 0.0
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["sequence"].to(device), batch["mask"].to(device))
            truth = batch["label"].to(device)
            total_loss += float(F.cross_entropy(logits, truth, reduction="sum"))
            probabilities = logits.softmax(dim=1).cpu().numpy()
            calc.update(logits.cpu().numpy(), truth.cpu().numpy())
            for path, true, prob in zip(batch["path"], truth.cpu().tolist(), probabilities):
                predicted = int(np.argmax(prob))
                rows.append({"path": path, "true": inverse[int(true)],
                             "predicted": inverse[predicted], "confidence": float(prob[predicted])})
    result = calc.compute()
    report = {"split": args.split, "checkpoint": str(args.checkpoint),
              "feature_mode": checkpoint.get("feature_mode", "full"),
              "loss": total_loss / len(dataset), "top1": result.top1_accuracy,
              "f1_macro": result.f1_macro, "total_samples": result.total_samples,
              "classes": label_map, "confusion_matrix": result.confusion_matrix.tolist(),
              "per_class_accuracy": result.per_class_accuracy}
    (output / f"{args.split}_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / f"{args.split}_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "true", "predicted", "confidence"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
