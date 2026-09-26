"""Módulo de treinamento: losses, métricas, trainer."""
from ml.training.losses import create_loss, compute_class_weights, LabelSmoothingCrossEntropy, FocalLoss
from ml.training.metrics import MetricsCalculator, MetricsResult
from ml.training.trainer import Trainer, TrainingConfig
