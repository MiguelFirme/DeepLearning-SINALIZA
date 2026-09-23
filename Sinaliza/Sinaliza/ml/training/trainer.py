"""
Treinador profissional para modelos de Libras.

Features:
    - AdamW com weight decay
    - Schedulers: CosineAnnealingWarmRestarts, ReduceLROnPlateau
    - Warmup linear
    - Early stopping por val_loss ou val_f1
    - Mixed precision (AMP)
    - Gradient clipping
    - Gradient accumulation
    - Checkpointing (melhor e último)
    - Logging CSV + JSON
"""
from __future__ import annotations
import csv
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.training.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    # Otimizador
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)

    # Scheduler
    scheduler_type: str = "cosine"  # cosine | plateau | none
    cosine_t0: int = 10
    cosine_t_mult: int = 2
    plateau_patience: int = 5
    plateau_factor: float = 0.5
    min_lr: float = 1e-7

    # Warmup
    warmup_epochs: int = 3

    # Treinamento
    epochs: int = 100
    gradient_clip_val: float = 1.0
    gradient_accumulation_steps: int = 1
    use_amp: bool = True

    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    monitor: str = "val_loss"  # val_loss | val_f1
    mode: str = "min"  # min (loss) | max (f1)

    # Checkpointing
    save_dir: str = "experiments"
    experiment_name: str = "default"
    save_top_k: int = 1


class EarlyStopping:
    """Early stopping com monitoramento de métrica."""

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value: float | None = None
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping! Sem melhora por {self.patience} épocas.")

        return self.should_stop


class Trainer:
    """Treinador completo para classificação de sinais."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        num_classes: int,
        config: TrainingConfig | None = None,
        label_names: dict[int, str] | None = None,
        device: str | torch.device | None = None,
    ):
        self.config = config or TrainingConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.num_classes = num_classes

        # Otimizador
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )

        # Scheduler
        self.scheduler = self._build_scheduler()

        # AMP
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Early stopping
        if self.config.early_stopping:
            self.early_stop = EarlyStopping(
                patience=self.config.patience,
                mode=self.config.mode,
            )
        else:
            self.early_stop = None

        # Métricas
        self.metrics_calc = MetricsCalculator(num_classes, label_names)

        # Diretórios
        self.save_dir = Path(self.config.save_dir) / self.config.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Histórico
        self.history: list[dict] = []
        self.best_metric: float | None = None
        self.best_epoch: int = 0

    def _build_scheduler(self):
        if self.config.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.config.cosine_t0, T_mult=self.config.cosine_t_mult,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=self.config.plateau_factor,
                patience=self.config.plateau_patience, min_lr=self.config.min_lr,
            )
        return None

    def _warmup_lr(self, epoch: int):
        """Aplica warmup linear."""
        if epoch < self.config.warmup_epochs:
            warmup_factor = (epoch + 1) / self.config.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.config.learning_rate * warmup_factor

    def train(self) -> dict:
        """Loop principal de treinamento."""
        logger.info(f"Iniciando treinamento: {self.config.epochs} épocas no dispositivo {self.device}")
        logger.info(f"Modelo: {self.model.__class__.__name__} | Params: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            self._warmup_lr(epoch)

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_data = {
                "epoch": epoch + 1,
                "lr": current_lr,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.history.append(epoch_data)

            # Scheduler step
            if epoch >= self.config.warmup_epochs and self.scheduler is not None:
                if self.config.scheduler_type == "plateau":
                    self.scheduler.step(val_metrics["loss"])
                elif self.config.scheduler_type == "cosine":
                    self.scheduler.step()

            # Checkpoint
            is_best = self._check_best(val_metrics)
            if is_best:
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            self._save_checkpoint(epoch, val_metrics, is_best=False)

            # Log
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_top1={val_metrics.get('top1', 0):.4f} | "
                f"val_f1={val_metrics.get('f1_macro', 0):.4f} | "
                f"lr={current_lr:.2e}"
                + (" ★" if is_best else "")
            )

            # Early stopping
            monitor_val = val_metrics["loss"] if self.config.monitor == "val_loss" else val_metrics.get("f1_macro", 0)
            if self.early_stop and self.early_stop.step(monitor_val):
                logger.info(f"Early stopping na época {epoch+1}.")
                break

        elapsed = time.time() - start_time
        self._save_history()

        result = {
            "best_epoch": self.best_epoch + 1,
            "best_metric": self.best_metric,
            "total_epochs": len(self.history),
            "elapsed_seconds": elapsed,
            "save_dir": str(self.save_dir),
        }
        logger.info(f"Treinamento concluído em {elapsed:.0f}s. Melhor época: {self.best_epoch+1}")
        return result

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            seq = batch["sequence"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(seq, mask)
                loss = self.criterion(logits, labels)
                loss = loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps * labels.size(0)
            total_samples += labels.size(0)

        return {"loss": total_loss / max(total_samples, 1)}

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        self.metrics_calc.reset()

        for batch in self.val_loader:
            seq = batch["sequence"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(seq, mask)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            self.metrics_calc.update(logits.cpu().numpy(), labels.cpu().numpy())

        metrics = self.metrics_calc.compute()
        return {
            "loss": total_loss / max(total_samples, 1),
            "top1": metrics.top1_accuracy,
            "top3": metrics.top3_accuracy,
            "top5": metrics.top5_accuracy,
            "f1_macro": metrics.f1_macro,
            "f1_weighted": metrics.f1_weighted,
            "precision_macro": metrics.precision_macro,
            "recall_macro": metrics.recall_macro,
        }

    def _check_best(self, val_metrics: dict) -> bool:
        if self.config.monitor == "val_loss":
            current = val_metrics["loss"]
            if self.best_metric is None or current < self.best_metric:
                self.best_metric = current
                self.best_epoch = len(self.history) - 1
                return True
        else:
            current = val_metrics.get("f1_macro", 0)
            if self.best_metric is None or current > self.best_metric:
                self.best_metric = current
                self.best_epoch = len(self.history) - 1
                return True
        return False

    def _save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool):
        name = "best.pt" if is_best else "last.pt"
        path = self.save_dir / name
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_metrics": val_metrics,
            "config": asdict(self.config),
            "model_class": self.model.__class__.__name__,
            "model_type": getattr(self.model, "sinaliza_model_type", None),
            "model_kwargs": getattr(self.model, "sinaliza_model_kwargs", {}),
            "target_frames": getattr(self.model, "sinaliza_target_frames", 48),
            "num_classes": self.num_classes,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, path)

    def _save_history(self):
        # CSV
        csv_path = self.save_dir / "history.csv"
        if self.history:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
                writer.writeheader()
                writer.writerows(self.history)

        # JSON
        json_path = self.save_dir / "history.json"
        with open(json_path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Histórico salvo em {csv_path} e {json_path}")
