"""
Funções de perda para treinamento.

Inclui:
    - CrossEntropy padrão com label smoothing e pesos de classe
    - Focal Loss para desbalanceamento extremo
    - Cálculo automático de pesos de classe (inversão de frequência)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(labels: list[int] | np.ndarray, num_classes: int, strategy: str = "inverse") -> torch.Tensor:
    """
    Calcula pesos de classe para lidar com desbalanceamento.

    Args:
        labels: lista de labels inteiros
        num_classes: número total de classes
        strategy: 'inverse' (1/freq), 'sqrt_inverse' (1/sqrt(freq)), 'effective' (1-β^n)

    Returns:
        Tensor de pesos (num_classes,) normalizado
    """
    counts = np.bincount(np.array(labels, dtype=int), minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)

    if strategy == "inverse":
        weights = 1.0 / counts
    elif strategy == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts)
    elif strategy == "effective":
        beta = 0.999
        weights = (1 - beta) / (1 - np.power(beta, counts))
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")

    # Normalizar para que a média seja 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class LabelSmoothingCrossEntropy(nn.Module):
    """CrossEntropy com label smoothing e pesos de classe opcionais."""

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C)
            targets: (B,) long
        """
        C = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Componente NLL
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        # Componente smooth
        smooth = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth

        if self.weight is not None:
            w = self.weight.to(logits.device)
            sample_weights = w[targets]
            loss = loss * sample_weights

        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss: penaliza menos exemplos fáceis.

    FL(pt) = -α * (1-pt)^γ * log(pt)
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = logits.size(-1)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # One-hot com smoothing
        if self.label_smoothing > 0:
            one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(-1), 1)
            one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / C
        else:
            one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(-1), 1)

        pt = (probs * one_hot).sum(dim=-1)  # probabilidade da classe correta
        focal_weight = (1 - pt) ** self.gamma
        loss = -(focal_weight * (one_hot * log_probs).sum(dim=-1))

        if self.weight is not None:
            w = self.weight.to(logits.device)
            sample_weights = w[targets]
            loss = loss * sample_weights

        return loss.mean()


def create_loss(
    loss_type: str = "cross_entropy",
    label_smoothing: float = 0.1,
    focal_gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """Factory para criar função de perda."""
    if loss_type == "cross_entropy":
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=class_weights)
    elif loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights, label_smoothing=label_smoothing)
    elif loss_type == "ce_standard":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Loss desconhecida: {loss_type}")
