#!/usr/bin/env python3
"""Compara dois checkpoints de 15 classes e sua fusão sequencial offline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data.dataset import LibrasDataset
from ml.models import create_model
from ml.training.metrics import MetricsCalculator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fuse_probabilities(
    pose: np.ndarray, full: np.ndarray, *, first: str, top_k: int,
    pose_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna escores com classes excluídas em -inf e candidatos da etapa 1."""
    if pose.shape != full.shape or pose.ndim != 2:
        raise ValueError("Probabilidades dos modelos precisam ter formato (vídeos, classes) idêntico")
    if first not in ("pose_face", "full"):
        raise ValueError("first deve ser pose_face ou full")
    if not 1 <= top_k <= pose.shape[1] or not 0 <= pose_weight <= 1:
        raise ValueError("top_k ou pose_weight inválido")
    first_probs = pose if first == "pose_face" else full
    candidates = np.argsort(-first_probs, axis=1, kind="stable")[:, :top_k]
    combined = pose_weight * pose + (1 - pose_weight) * full
    scores = np.full(combined.shape, -np.inf, dtype=np.float64)
    np.put_along_axis(scores, candidates, np.take_along_axis(combined, candidates, axis=1), axis=1)
    return scores, candidates


def metrics(scores: np.ndarray, truth: np.ndarray, names: dict[int, str]) -> dict:
    calc = MetricsCalculator(len(names), names)
    calc.update(scores, truth)
    result = calc.compute()
    return {
        "correct": int((scores.argmax(axis=1) == truth).sum()),
        "total": len(truth),
        "top1": result.top1_accuracy,
        "f1_macro": result.f1_macro,
        "confusion_matrix": result.confusion_matrix.tolist(),
        "per_class_accuracy": {names[k]: v for k, v in result.per_class_accuracy.items()},
    }


def infer(checkpoint_path: Path, data_dir: Path, label_map: dict[str, int],
          indices: list[int], device: torch.device, expected_mode: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mode = checkpoint.get("feature_mode", "full")
    if mode != expected_mode:
        raise ValueError(f"{checkpoint_path}: esperava feature_mode={expected_mode}, recebi {mode}")
    if int(checkpoint.get("target_frames", 48)) != 48:
        raise ValueError(f"{checkpoint_path}: este experimento requer 48 frames")
    dataset = LibrasDataset(data_dir, label_map=label_map, target_frames=48,
                            split_indices=indices, feature_mode=mode)
    if len(dataset) != len(indices):
        raise ValueError("O dataset perdeu amostras do split; confira manifesto e mapa de classes")
    model_type = checkpoint.get("model_type") or checkpoint["model_class"].lower().replace("model", "")
    model = create_model(model_type, input_dim=dataset.feature_dim,
                         num_classes=len(label_map), **checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    paths, labels, probabilities = [], [], []
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["sequence"].to(device), batch["mask"].to(device))
            paths.extend(batch["path"])
            labels.extend(batch["label"].tolist())
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
    return paths, np.asarray(labels, dtype=np.int64), np.concatenate(probabilities, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-checkpoint", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--pose-split-dir", type=Path, required=True)
    parser.add_argument("--full-split-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--first", choices=("pose_face", "full"), default="pose_face")
    parser.add_argument("--top-k", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--pose-weight", type=float, default=0.65)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for name in ("label_map.json", "splits.json", "manifest.json"):
        pose_obj = json.loads((args.pose_split_dir / name).read_text(encoding="utf-8"))
        full_obj = json.loads((args.full_split_dir / name).read_text(encoding="utf-8"))
        if pose_obj != full_obj:
            raise ValueError(f"Os arquivos {name} dos dois checkpoints diferem")
    label_map = json.loads((args.pose_split_dir / "label_map.json").read_text(encoding="utf-8"))
    splits = json.loads((args.pose_split_dir / "splits.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.pose_split_dir / "manifest.json").read_text(encoding="utf-8"))
    current_manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    if current_manifest != manifest:
        raise ValueError("O manifesto dos .npz difere do manifesto arquivado; índices não são seguros")
    if sorted(label_map.values()) != list(range(len(label_map))):
        raise ValueError("label_map deve conter índices contíguos de zero a N-1")
    indices = splits[args.split]
    if len(indices) != len(set(indices)) or any(
        set(indices) & set(splits[other]) for other in ("train", "val", "test") if other != args.split
    ):
        raise ValueError("O split possui duplicações ou vazamento com outro split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_paths, pose_truth, pose = infer(args.pose_checkpoint, args.data_dir, label_map, indices,
                                        device, "pose_face")
    full_paths, full_truth, full = infer(args.full_checkpoint, args.data_dir, label_map, indices,
                                        device, "full")
    if pose_paths != full_paths or not np.array_equal(pose_truth, full_truth):
        raise ValueError("Predições dos modelos não estão alinhadas por vídeo e classe")
    inverse = {value: key for key, value in label_map.items()}
    report = {
        "split": args.split, "first": args.first, "pose_weight": args.pose_weight,
        "full_weight": 1 - args.pose_weight, "device": str(device),
        "pose_checkpoint": str(args.pose_checkpoint),
        "pose_sha256": sha256(args.pose_checkpoint),
        "full_checkpoint": str(args.full_checkpoint),
        "full_sha256": sha256(args.full_checkpoint),
        "label_map": label_map, "split_size": len(indices),
        "pose_face": metrics(pose, pose_truth, inverse),
        "full": metrics(full, pose_truth, inverse), "fusion": {},
    }
    rows = []
    for top_k in args.top_k:
        scores, candidates = fuse_probabilities(pose, full, first=args.first,
                                                 top_k=top_k, pose_weight=args.pose_weight)
        predicted = scores.argmax(axis=1)
        coverage = int(np.any(candidates == pose_truth[:, None], axis=1).sum())
        report["fusion"][str(top_k)] = {
            **metrics(scores, pose_truth, inverse),
            "candidate_coverage_count": coverage,
            "candidate_coverage": coverage / len(pose_truth),
        }
        for i, path in enumerate(pose_paths):
            choice = int(predicted[i])
            rows.append({
                "top_k": top_k, "path": path, "true": inverse[int(pose_truth[i])],
                "pose_prediction": inverse[int(pose[i].argmax())],
                "full_prediction": inverse[int(full[i].argmax())],
                "candidates": json.dumps([inverse[int(c)] for c in candidates[i]], ensure_ascii=False),
                "truth_in_candidates": bool(pose_truth[i] in candidates[i]),
                "fused_prediction": inverse[choice], "fused_correct": bool(choice == pose_truth[i]),
                "fused_score": float(scores[i, choice]),
                "pose_probabilities": json.dumps({inverse[j]: float(p) for j, p in enumerate(pose[i])}, ensure_ascii=False),
                "full_probabilities": json.dumps({inverse[j]: float(p) for j, p in enumerate(full[i])}, ensure_ascii=False),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_{args.first}"
    (args.output_dir / f"{stem}_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / f"{stem}_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"split": args.split, "first": args.first,
                      "pose_face": {k: report["pose_face"][k] for k in ("correct", "top1", "f1_macro")},
                      "full": {k: report["full"][k] for k in ("correct", "top1", "f1_macro")},
                      "fusion": {k: {m: v[m] for m in ("correct", "top1", "f1_macro", "candidate_coverage_count")}
                                 for k, v in report["fusion"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
