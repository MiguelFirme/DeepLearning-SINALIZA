"""Orquestração de treino v2; features compartilhadas com a webcam."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from ml.data.augmentation_v2 import RawAugmentorV2, AugmentationV2Config
from ml.data.dataset_v2 import LibrasDatasetV2
from ml.features.pipeline_v2 import PipelineV2Config, FEATURE_DIM_V2
from ml.models import create_model
from ml.training.trainer import Trainer, TrainingConfig

logger = logging.getLogger(__name__)


def validate_splits_v2(manifest, splits, labels):
    """Recusa vídeo/pessoa em múltiplos conjuntos e classes sem treino."""
    indices = [i for split in splits.values() for i in split]
    if len(indices) != len(set(indices)) or any(not isinstance(i, int) or not 0 <= i < len(manifest) for i in indices):
        raise ValueError("Índices inválidos ou repetidos entre splits")
    people, videos = {}, {}
    for split, values in splits.items():
        for i in values:
            row = manifest[i]
            if row["label"] not in labels:
                raise ValueError("Classe sem rótulo no split")
            person = row.get("person") or row.get("signer")
            if not person or person == "unknown":
                raise ValueError("V2 exige identificação da pessoa para impedir vazamento")
            video = row.get("source_video") or row.get("video_id") or row["path"]
            for key, owners in ((person, people), (video, videos)):
                if key in owners and owners[key] != split:
                    raise ValueError("Pessoa ou vídeo presente em mais de um split")
                owners[key] = split
            if split != "train" and row.get("is_synthetic"):
                raise ValueError("Validação/teste não podem conter derivados sintéticos")
    if not splits.get("train") or not splits.get("val"):
        raise ValueError("V2 exige treino e validação independentes por pessoa")
    if set(labels.values()) != set(range(len(labels))):
        raise ValueError("Mapa de rótulos precisa ser contíguo")
    if set(labels) - {manifest[i]["label"] for i in splits["train"]}:
        raise ValueError("Classe sem exemplo no treino")


def balanced_sampler(labels, seed):
    """Pesos calculados exclusivamente nos rótulos do treino."""
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels)
    weights = 1 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(labels),
                                 replacement=True, generator=torch.Generator().manual_seed(seed))


def run_training_v2(args, cfg):
    """Ponto de entrada do CLI de treino quando pipeline_version=2."""
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pipeline = PipelineV2Config(**cfg.get("pipeline", {}))
    if args.target_frames and args.target_frames != pipeline.n_samples:
        raise ValueError("target_frames precisa coincidir com FPS × duração v2")
    data_dir, processed = Path(args.data_dir), Path(args.processed_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest != json.loads((processed / "manifest.json").read_text(encoding="utf-8")):
        raise ValueError("Manifesto de dados diferente do manifesto dos splits")
    splits = json.loads((processed / "splits.json").read_text(encoding="utf-8"))
    labels = json.loads((processed / "label_map.json").read_text(encoding="utf-8"))
    validate_splits_v2(manifest, splits, labels)
    aug_mapping = cfg.get("augmentation", {})
    if cfg.get("augmentation_file"):
        aug_mapping = yaml.safe_load((Path(args.config).parent / cfg["augmentation_file"]).read_text(encoding="utf-8"))
    aug = RawAugmentorV2(AugmentationV2Config(**aug_mapping))
    datasets = {split: LibrasDatasetV2(data_dir, label_map=labels, split_indices=indices, cfg=pipeline,
                                      training=split == "train", augmentor=aug if split == "train" else None,
                                      augmentation_seed=seed)
                for split, indices in splits.items() if indices}
    train_cfg = cfg.get("training", {})
    if args.num_workers not in (None, 0) or cfg.get("data", {}).get("num_workers", 0):
        logger.info("Workers têm sementes distintas por época; persistent_workers permanece desativado")
    sampler = balanced_sampler(datasets["train"].get_labels(), seed) if train_cfg.get("balanced_sampler", True) else None
    batch_size = args.batch_size or train_cfg.get("batch_size", 16)
    workers = args.num_workers if args.num_workers is not None else cfg.get("data", {}).get("num_workers", 0)
    loaders = {split: DataLoader(ds, batch_size=batch_size, num_workers=workers,
                                 shuffle=split == "train" and sampler is None,
                                 sampler=sampler if split == "train" else None)
               for split, ds in datasets.items()}
    model_type = args.model or cfg.get("model", {}).get("type", "transformer")
    kwargs = cfg.get("model", {}).get("params", {}).copy()
    if kwargs.get("max_seq_len", pipeline.n_samples) < pipeline.n_samples:
        raise ValueError("Modelo não comporta a janela temporal configurada")
    model = create_model(model_type, input_dim=FEATURE_DIM_V2, num_classes=len(labels), **kwargs)
    model.sinaliza_model_type, model.sinaliza_model_kwargs = model_type, kwargs
    model.sinaliza_target_frames, model.sinaliza_feature_mode = pipeline.n_samples, "full"
    model.sinaliza_pipeline_metadata = {"pipeline_version": 2, "pipeline_config": asdict(pipeline),
                                         "feature_dim": FEATURE_DIM_V2, "label_map": labels,
                                         "target_fps": pipeline.target_fps, "window_seconds": pipeline.window_seconds}
    name = args.experiment or "transformer_v2_mixed25"
    config = TrainingConfig(optimizer_type=train_cfg.get("optimizer", "adamw"),
                            learning_rate=args.lr or train_cfg.get("learning_rate", .001),
                            weight_decay=train_cfg.get("weight_decay", .01),
                            scheduler_type=train_cfg.get("scheduler", "cosine"),
                            warmup_epochs=train_cfg.get("warmup_epochs", 5),
                            epochs=args.epochs or train_cfg.get("epochs", 150),
                            patience=train_cfg.get("patience", 25), monitor="val_f1", mode="max",
                            save_dir=args.save_dir, experiment_name=name, use_amp=train_cfg.get("use_amp", True))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing if args.label_smoothing is not None
                                         else train_cfg.get("label_smoothing", .1))
    log_dir = Path(args.log_dir) / Path(args.save_dir).name / name
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(handler)
    record = {"status": "running", "started_at": datetime.now().isoformat(), "config": cfg,
              "training": asdict(config), "pipeline": asdict(pipeline), "seed": seed,
              "metadata_issues": {split: ds.metadata_issues for split, ds in datasets.items()},
              "hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                         (data_dir / "manifest.json", processed / "splits.json", processed / "label_map.json")}}
    record_path = log_dir / "run_record.json"
    def save_record():
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    save_record()
    trainer = Trainer(model, loaders["train"], loaders["val"], criterion, len(labels), config,
                      {i: word for word, i in labels.items()}, device)
    try:
        result = trainer.train()
        trainer.load_best_weights()
        for split in ("val", "test"):
            if split not in loaders or (split == "test" and args.skip_test):
                continue
            if split == "test" and args.test_checkpoint == "last":
                trainer.load_last_weights()
            metrics = trainer.evaluate_loader(loaders[split])
            (trainer.save_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("%s — métricas por origem e matriz: %s", split, json.dumps(metrics, ensure_ascii=False))
        record.update(status="completed", result=result)
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        record["finished_at"] = datetime.now().isoformat()
        save_record()
        logging.getLogger().removeHandler(handler)
        handler.close()
    return result
