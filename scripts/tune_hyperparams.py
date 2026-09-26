#!/usr/bin/env python3
"""
Otimização de hiperparâmetros com Optuna.

Uso:
    python scripts/tune_hyperparams.py --model bilstm --n-trials 50
"""
import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_objective(model_type, train_ds, val_ds, num_classes, input_dim, device):
    """Cria função objetivo para Optuna."""

    def objective(trial):
        import optuna
        from ml.models import create_model
        from ml.training.losses import create_loss, compute_class_weights
        from ml.training.trainer import Trainer, TrainingConfig

        # Hiperparâmetros
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

        if model_type == "bilstm":
            hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
            num_layers = trial.suggest_int("num_layers", 1, 3)
            model_kwargs = {"hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout}
        elif model_type == "transformer":
            d_model = trial.suggest_categorical("d_model", [128, 256, 512])
            nhead = trial.suggest_categorical("nhead", [4, 8])
            num_layers = trial.suggest_int("num_layers", 2, 6)
            model_kwargs = {"d_model": d_model, "nhead": nhead, "num_layers": num_layers, "dropout": dropout}
        else:
            num_channels = trial.suggest_categorical("num_channels", [128, 256])
            num_blocks = trial.suggest_int("num_blocks", 4, 8)
            model_kwargs = {"num_channels": num_channels, "num_blocks": num_blocks, "dropout": dropout}

        model = create_model(model_type, input_dim=input_dim, num_classes=num_classes, **model_kwargs)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        class_weights = compute_class_weights(train_ds.get_labels(), num_classes, strategy="sqrt_inverse")
        criterion = create_loss("cross_entropy", label_smoothing=label_smoothing, class_weights=class_weights)

        config = TrainingConfig(
            learning_rate=lr,
            weight_decay=weight_decay,
            epochs=30,  # épocas reduzidas para tuning
            patience=8,
            experiment_name=f"optuna_trial_{trial.number}",
            save_dir="experiments/optuna",
        )

        trainer = Trainer(model, train_loader, val_loader, criterion, num_classes, config=config, device=device)
        result = trainer.train()

        return trainer.best_metric  # val_loss (minimizar)

    return objective


def main():
    parser = argparse.ArgumentParser(description="Tuning com Optuna")
    parser.add_argument("--model", type=str, default="bilstm")
    parser.add_argument("--data-dir", type=str, default="data/landmarks")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--target-frames", type=int, default=48)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        logger.error("Instale optuna: pip install optuna")
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processed_dir = Path(args.processed_dir)

    with open(processed_dir / "label_map.json") as f:
        label_map = json.load(f)
    with open(processed_dir / "splits.json") as f:
        splits = json.load(f)

    num_classes = len(label_map)
    from ml.data.dataset import LibrasDataset
    from ml.data.augmentation import SequenceAugmentor, AugmentationConfig

    data_dir = Path(args.data_dir)
    augmentation_config = AugmentationConfig(enabled=False)
    aug = SequenceAugmentor(augmentation_config) if augmentation_config.enabled else None
    train_ds = LibrasDataset(data_dir, label_map=label_map, target_frames=args.target_frames,
                             augmentor=aug, split_indices=splits["train"])
    val_ds = LibrasDataset(data_dir, label_map=label_map, target_frames=args.target_frames,
                           split_indices=splits["val"])

    input_dim = train_ds.feature_dim
    objective = create_objective(args.model, train_ds, val_ds, num_classes, input_dim, device)

    study = optuna.create_study(direction="minimize", study_name=f"sinaliza_{args.model}")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    logger.info(f"Melhor trial: {study.best_trial.number}")
    logger.info(f"Melhor val_loss: {study.best_value:.4f}")
    logger.info(f"Melhores params: {json.dumps(study.best_params, indent=2)}")

    # Salvar resultados
    output_dir = Path("experiments/optuna")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"best_params_{args.model}.json", "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)


if __name__ == "__main__":
    main()
