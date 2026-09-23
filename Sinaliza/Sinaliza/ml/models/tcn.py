"""
Modelo C — Temporal Convolutional Network (TCN) para classificação de sinais.

Arquitetura: Projeção Linear → N blocos residuais TCN (conv1d dilatada causal)
             → Global Average + Max Pooling → MLP Head.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from ml.models.base import BaseSignModel


class CausalConv1d(nn.Module):
    """Convolução 1D causal com padding à esquerda."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Bloco residual TCN: 2 × (CausalConv → BatchNorm → GELU → Dropout) + skip."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNModel(BaseSignModel):
    """TCN com blocos residuais dilatados e pooling global."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_channels: int = 256,
        num_blocks: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.2,
        classifier_hidden: int = 256,
        **kwargs,
    ):
        super().__init__(input_dim, num_classes)
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, num_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            TCNBlock(num_channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(num_blocks)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_channels * 2),  # avg + max pooling concatenados
            nn.Dropout(dropout),
            nn.Linear(num_channels * 2, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            mask: (B, T) bool
        Returns:
            logits: (B, num_classes)
        """
        x = self.input_norm(x)
        x = self.projection(x)  # (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)

        for block in self.blocks:
            x = block(x)

        # Mascarar frames inválidos antes do pooling
        if mask is not None:
            m = mask.unsqueeze(1).float()  # (B, 1, T)
            x = x * m
            avg = x.sum(dim=2) / m.sum(dim=2).clamp(min=1)
            x_masked = x.masked_fill(m == 0, float("-inf"))
            mx = x_masked.max(dim=2).values
            mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
        else:
            avg = x.mean(dim=2)
            mx = x.max(dim=2).values

        pooled = torch.cat([avg, mx], dim=-1)  # (B, 2C)
        return self.classifier(pooled)
