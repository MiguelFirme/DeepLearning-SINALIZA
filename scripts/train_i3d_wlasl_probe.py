#!/usr/bin/env python3
"""Linear probe I3D RGB/WLASL2000 nos videos MINDS-Libras.

Carrega pesos oficiais, extrai embeddings congelados e ajusta apenas LogisticRegression.
Usa somente train/val do split existente; nunca le os videos de test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("i3d_wlasl_probe")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_splits(manifest_path: Path, split_path: Path,
                label_map_path: Path) -> tuple[dict, dict[str, int]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    labels = json.loads(label_map_path.read_text(encoding="utf-8"))
    selected = {}
    for name in ("train", "val", "test"):
        ids = splits[name]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Indices duplicados em {name}")
        selected[name] = [manifest[i] for i in ids]
    groups = {name: {s["signer"] for s in samples} for name, samples in selected.items()}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[first] & groups[second]:
            raise ValueError(f"Vazamento de sinalizador entre {first} e {second}")
    for name in ("train", "val"):
        if {s["label"] for s in selected[name]} != set(labels):
            raise ValueError(f"Classes incompletas em {name}")
        if len({s["video_id"] for s in selected[name]}) != len(selected[name]):
            raise ValueError(f"Video repetido em {name}")
    LOG.info("Train %d %s | Val %d %s | Test reservado %d %s",
             len(selected["train"]), sorted(groups["train"]),
             len(selected["val"]), sorted(groups["val"]),
             len(selected["test"]), sorted(groups["test"]))
    return selected, labels


def read_video_tensor(path: Path, frames: int = 16, size: int = 224) -> torch.Tensor:
    """Amostragem uniforme, recorte central quadrado e RGB [-1,1]."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video nao abriu: {path}")
    try:
        decoded = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            height, width = image.shape[:2]
            square = min(height, width)
            x0 = (width - square) // 2
            y0 = (height - square) // 2
            image = image[y0:y0 + square, x0:x0 + square]
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            decoded.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not decoded:
            raise ValueError(f"Video sem frames decodificaveis: {path}")
        # CAP_PROP_FRAME_COUNT pode diferir da contagem realmente decodificada.
        indices = np.linspace(0, len(decoded) - 1, frames, dtype=int)
        video = np.stack([decoded[int(i)] for i in indices]).astype(np.float32)
        video = video / 127.5 - 1.0
        return torch.from_numpy(video).permute(3, 0, 1, 2).contiguous()
    finally:
        cap.release()


def load_i3d(source: Path, checkpoint: Path, device: torch.device):
    spec = importlib.util.spec_from_file_location("official_wlasl_pytorch_i3d", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel importar {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = module.InceptionI3d(2000, in_channels=3)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    LOG.info("I3D WLASL2000 carregado integralmente: %d tensores", len(state))
    return model


def extract_split(samples: list[dict], video_dir: Path, cache_dir: Path,
                  model, device: torch.device) -> np.ndarray:
    vectors = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for position, sample in enumerate(samples, 1):
        video_id = sample["video_id"]
        cache_path = cache_dir / f"{Path(video_id).stem}.npy"
        if cache_path.is_file():
            vector = np.load(cache_path, allow_pickle=False)
        else:
            tensor = read_video_tensor(video_dir / video_id).unsqueeze(0).to(device)
            with torch.inference_mode():
                vector = model.extract_features(tensor).mean(dim=(2, 3, 4)).squeeze(0)
            vector = vector.cpu().numpy().astype(np.float32)
            np.save(cache_path, vector)
        if vector.shape != (1024,) or not np.isfinite(vector).all():
            raise ValueError(f"Embedding invalido: {video_id}, {vector.shape}")
        vectors.append(vector)
        if position % 10 == 0 or position == len(samples):
            LOG.info("Embeddings: %d/%d", position, len(samples))
    return np.stack(vectors)


def metrics(y_true: np.ndarray, probs: np.ndarray, classes: np.ndarray) -> dict:
    prediction = classes[np.argmax(probs, axis=1)]
    return {
        "top1": float(accuracy_score(y_true, prediction)),
        "f1_macro": float(f1_score(y_true, prediction, labels=classes,
                                   average="macro", zero_division=0)),
        "loss": float(log_loss(y_true, probs, labels=classes)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=classes).tolist(),
        "total_samples": len(y_true),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/wlasl_source/code/I3D/pytorch_i3d.py"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/wlasl_asl2000.pt"))
    parser.add_argument("--video-dir", type=Path, default=Path("data/raw/minds_libras"))
    parser.add_argument("--manifest", type=Path, default=Path("data/landmarks/minds_libras_school5/manifest.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/minds_libras_school5/splits.json"))
    parser.add_argument("--label-map", type=Path, default=Path("data/processed/minds_libras_school5/label_map.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/i3d_wlasl_school5_embeddings"))
    parser.add_argument("--output", type=Path, default=Path("experiments/i3d_wlasl_school5_seed42"))
    parser.add_argument("--limit-per-split", type=int, help="Somente extracao de embeddings para smoke test")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (Path("logs/experiments") / args.output.name).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            Path("logs/experiments") / args.output.name / "train.log", encoding="utf-8")])
    start = time.time()
    np.random.seed(42)
    torch.manual_seed(42)
    selected, label_map = load_splits(args.manifest, args.splits, args.label_map)
    checkpoint_hash = sha256(args.checkpoint)
    source_hash = sha256(args.source)
    identity = {"checkpoint_sha256": checkpoint_hash, "source_sha256": source_hash,
                "manifest_sha256": sha256(args.manifest), "splits_sha256": sha256(args.splits),
                "label_map_sha256": sha256(args.label_map), "frames": 16,
                "image_size": 224, "sampling": "uniform_full_video",
                "crop": "center_square", "pixels": "RGB_-1_to_1"}
    meta_path = args.cache_dir / "cache_metadata.json"
    if meta_path.is_file():
        if json.loads(meta_path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Cache pertence a outra configuracao: {meta_path}")
    else:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        if list(args.cache_dir.glob("*.npy")):
            raise ValueError("Cache sem metadados: escolha outra pasta")
        meta_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Dispositivo %s | checkpoint SHA256 %s", device, checkpoint_hash)
    model = load_i3d(args.source, args.checkpoint, device)
    x, y = {}, {}
    for name in ("train", "val"):
        samples = selected[name]
        if args.limit_per_split is not None:
            samples = samples[:args.limit_per_split]
        x[name] = extract_split(samples, args.video_dir, args.cache_dir, model, device)
        y[name] = np.array([label_map[s["label"]] for s in samples], dtype=int)
    if args.limit_per_split is not None:
        LOG.info("Smoke test concluido: sem treino ou avaliacao")
        return
    del model
    if set(np.unique(y["train"])) != set(label_map.values()):
        raise ValueError("Classes ausentes no treino")
    head = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=3000, random_state=42))
    head.fit(x["train"], y["train"])
    classes = head.classes_
    result = {name: metrics(y[name], head.predict_proba(x[name]), classes)
              for name in ("train", "val")}
    inverse = {value: key for key, value in label_map.items()}
    with (args.output / "val_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "signer", "true", "predicted", "confidence"])
        writer.writeheader()
        probs = head.predict_proba(x["val"])
        for sample, true, prob in zip(selected["val"], y["val"], probs):
            pred_idx = int(np.argmax(prob))
            writer.writerow({"video_id": sample["video_id"], "signer": sample["signer"],
                             "true": inverse[int(true)], "predicted": inverse[int(classes[pred_idx])],
                             "confidence": float(prob[pred_idx])})
    joblib.dump(head, args.output / "head.joblib")
    record = {"model": "I3D_RGB_WLASL2000_frozen_logistic_regression",
              "identity": identity, "classes": label_map, "train": result["train"],
              "val": result["val"], "test_evaluated": False,
              "duration_seconds": round(time.time() - start, 1),
              "torch": torch.__version__}
    (args.output / "run_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Resultado train %s | val %s", result["train"], result["val"])
    LOG.info("Registro: %s", args.output / "run_record.json")


if __name__ == "__main__":
    main()
