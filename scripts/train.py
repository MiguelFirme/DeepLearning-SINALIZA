#!/usr/bin/env python3
"""
Script principal de treinamento.

Uso:
    python scripts/train.py --config configs/bilstm.yaml
    python scripts/train.py --model bilstm --epochs 100 --lr 1e-3
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def merge_config(base: dict, override: dict) -> dict:
    """Combina configurações aninhadas sem descartar a configuração base."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | None) -> dict:
    """Carrega configuração YAML."""
    if config_path and Path(config_path).exists():
        import yaml
        path = Path(config_path)
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        base_name = config.pop("_base_", None)
        if base_name:
            return merge_config(load_config(str(path.parent / base_name)), config)
        return config
    return {}


def main():
    parser = argparse.ArgumentParser(description="Treinar modelo Sinaliza")
    parser.add_argument("--config", type=str, help="Arquivo YAML de configuração")
    parser.add_argument("--model", type=str, default=None, choices=["bilstm", "transformer", "tcn"])
    parser.add_argument("--data-dir", type=str, default="data/landmarks")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-frames", type=int, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="experiments")
    parser.add_argument("--loss", type=str, default=None, choices=["cross_entropy", "focal", "ce_standard"])
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Carregar config
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg_dict = cfg.get("training", {})
    model_type = args.model or cfg.get("model", {}).get("type", "bilstm")
    target_frames = args.target_frames or data_cfg.get("target_frames", 48)
    batch_size = args.batch_size or train_cfg_dict.get("batch_size", data_cfg.get("batch_size", 32))
    num_workers = args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 2)
    epochs = args.epochs or train_cfg_dict.get("epochs", 100)
    learning_rate = args.lr or train_cfg_dict.get("learning_rate", train_cfg_dict.get("lr", 1e-3))
    label_smoothing = (
        args.label_smoothing
        if args.label_smoothing is not None
        else train_cfg_dict.get("label_smoothing", 0.1)
    )
    loss_type = args.loss or train_cfg_dict.get("loss", "cross_entropy")
    if loss_type == "label_smoothing":
        loss_type = "cross_entropy"
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    experiment_name = args.experiment or f"{model_type}_{target_frames}f"

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Carregar splits e label_map
    processed_dir = Path(args.processed_dir)
    splits_path = processed_dir / "splits.json"
    label_map_path = processed_dir / "label_map.json"

    if not splits_path.exists() or not label_map_path.exists():
        logger.error("Execute build_dataset.py antes de treinar!")
        sys.exit(1)

    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)
    with open(label_map_path, encoding="utf-8") as f:
        label_map = json.load(f)
    with open(Path(args.data_dir) / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(processed_dir / "manifest.json", encoding="utf-8") as f:
        split_manifest = json.load(f)
    if manifest != split_manifest:
        raise ValueError("O manifesto de landmarks mudou após gerar os splits; reconstrua o dataset")
    all_indices = [i for indices in splits.values() for i in indices]
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("A mesma amostra aparece em mais de um split")
    for split_name, indices in splits.items():
        if any(not isinstance(i, int) or i < 0 or i >= len(manifest) for i in indices):
            raise ValueError(f"Índice inválido no split {split_name}")
        unknown = {manifest[i]["label"] for i in indices} - set(label_map)
        if unknown:
            raise ValueError(f"Classes do split {split_name} ausentes de label_map: {sorted(unknown)}")
    if not splits["train"]:
        raise ValueError("Split de treino vazio")
    missing_train = set(label_map) - {manifest[i]["label"] for i in splits["train"]}
    if missing_train:
        raise ValueError(f"Classes sem amostras de treino: {sorted(missing_train)}")

    num_classes = len(label_map)
    logger.info(f"Classes: {num_classes}")

    # Datasets
    from ml.data.dataset import LibrasDataset
    from ml.data.augmentation import SequenceAugmentor, AugmentationConfig
    from ml.features.sequence import SequenceConfig

    seq_cfg = SequenceConfig(target_frames=target_frames)
    augmentation_config = AugmentationConfig.from_mapping(cfg.get("augmentation", {}))
    aug = (SequenceAugmentor(augmentation_config, rng=np.random.default_rng(seed))
           if augmentation_config.enabled else None)

    data_dir = Path(args.data_dir)
    train_ds = LibrasDataset(data_dir, label_map=label_map, target_frames=target_frames,
                             seq_config=seq_cfg, augmentor=aug, split_indices=splits["train"])
    val_indices = splits.get("val", [])
    val_ds = None
    if val_indices:
        val_ds = LibrasDataset(
            data_dir,
            label_map=label_map,
            target_frames=target_frames,
            seq_config=seq_cfg,
            split_indices=val_indices,
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=device.type == "cuda", drop_last=False)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=device.type == "cuda")

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds) if val_ds is not None else 0}")

    # Modelo
    from ml.models import create_model
    input_dim = train_ds.feature_dim
    model_cfg = dict(cfg.get("model", {}).get("params", cfg.get("model", {})))
    for key in ("type", "input_dim", "num_classes"):
        model_cfg.pop(key, None)
    if model_type == "bilstm" and "hidden_dim" in model_cfg:
        model_cfg["hidden_size"] = model_cfg.pop("hidden_dim")
    if model_type == "transformer" and "max_len" in model_cfg:
        model_cfg["max_seq_len"] = model_cfg.pop("max_len")
    model = create_model(model_type, input_dim=input_dim, num_classes=num_classes, **model_cfg)
    model.sinaliza_model_type = model_type
    model.sinaliza_model_kwargs = model_cfg
    model.sinaliza_target_frames = target_frames
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Modelo: {model_type} | input_dim={input_dim} | params={param_count:,}")

    # Loss
    from ml.training.losses import create_loss, compute_class_weights
    weights_strategy = train_cfg_dict.get(
        "class_weight_method",
        train_cfg_dict.get("class_weights", "none"),
    )
    class_weights = None
    if weights_strategy not in (None, "none", "disabled"):
        class_weights = compute_class_weights(
            train_ds.get_labels(), num_classes, strategy=weights_strategy,
            effective_beta=train_cfg_dict.get("effective_beta", 0.999),
        )
    criterion = create_loss(
        loss_type, label_smoothing=label_smoothing, class_weights=class_weights,
        focal_gamma=(args.focal_gamma if args.focal_gamma is not None
                     else train_cfg_dict.get("focal_gamma", 2.0)),
    )
    logger.info("Loss=%s | class_weight_method=%s | smoothing=%s", loss_type,
                weights_strategy, label_smoothing)

    # Trainer
    from ml.training.trainer import Trainer, TrainingConfig
    train_config = TrainingConfig(
        learning_rate=learning_rate,
        epochs=epochs,
        weight_decay=train_cfg_dict.get("weight_decay", 1e-4),
        betas=tuple(train_cfg_dict.get("betas", (0.9, 0.999))),
        scheduler_type=train_cfg_dict.get("scheduler", "cosine"),
        warmup_epochs=train_cfg_dict.get("warmup_epochs", 3),
        patience=train_cfg_dict.get("patience", 15),
        early_stopping=train_cfg_dict.get("early_stopping", True),
        gradient_clip_val=train_cfg_dict.get("gradient_clip", 1.0),
        gradient_accumulation_steps=train_cfg_dict.get("accumulation_steps", 1),
        use_amp=train_cfg_dict.get("use_amp", True),
        experiment_name=experiment_name,
        save_dir=args.save_dir,
    )

    inv_label = {v: k for k, v in label_map.items()}
    trainer = Trainer(model, train_loader, val_loader, criterion, num_classes,
                      config=train_config, label_names=inv_label, device=device)

    # Treinar
    result = trainer.train()

    # O articulador retido não participa de nenhuma decisão durante as épocas.
    # Ele só é carregado e avaliado depois que o treino e a seleção terminaram.
    test_indices = splits.get("test", [])
    if test_indices:
        test_ds = LibrasDataset(
            data_dir,
            label_map=label_map,
            target_frames=target_frames,
            seq_config=seq_cfg,
            split_indices=test_indices,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        trainer.load_best_weights()
        test_metrics = trainer.evaluate_loader(test_loader)
        test_metrics_path = trainer.save_dir / "test_metrics.json"
        with open(test_metrics_path, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2, ensure_ascii=False)
        result["test_metrics"] = test_metrics
        result["test_metrics_path"] = str(test_metrics_path)
        logger.info(f"Teste final: {json.dumps(test_metrics, ensure_ascii=False)}")

    logger.info(f"Resultado: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
