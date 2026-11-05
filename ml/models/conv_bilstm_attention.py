"""Temporal Conv1D followed by a bidirectional LSTM and masked attention."""
from __future__ import annotations

import torch
from torch import nn

from ml.models.base import BaseSignModel
from ml.models.bilstm import TemporalAttention


class ConvBiLSTMAttentionModel(BaseSignModel):
    """Small sequence classifier for shoulder-normalized MediaPipe features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        projection_dim: int = 64,
        conv_channels: int = 64,
        conv_kernel_size: int = 3,
        hidden_size: int = 32,
        num_layers: int = 2,
        classifier_hidden: int = 64,
        dropout: float = 0.45,
    ):
        super().__init__(input_dim, num_classes)
        if conv_kernel_size < 1 or conv_kernel_size % 2 != 1:
            raise ValueError("conv_kernel_size must be a positive odd number")
        if num_layers < 1 or hidden_size < 1:
            raise ValueError("num_layers and hidden_size must be positive")

        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        self.conv = nn.Conv1d(
            projection_dim, conv_channels, conv_kernel_size,
            padding=conv_kernel_size // 2, bias=False,
        )
        self.conv_norm = nn.BatchNorm1d(conv_channels)
        self.conv_activation = nn.GELU()
        self.conv_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=conv_channels, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        output_dim = hidden_size * 2
        self.attention = TemporalAttention(output_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != self.input_dim:
            raise ValueError(f"Expected (batch, frames, {self.input_dim}), got {tuple(x.shape)}")
        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError("mask must have shape (batch, frames)")
            mask = mask.bool().clone()
            # An empty recording must not produce NaNs in masked softmax.
            mask[~mask.any(dim=1), 0] = True

        x = self.projection(self.input_norm(x))
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = x.transpose(1, 2)
        x = self.conv_dropout(self.conv_activation(self.conv_norm(self.conv(x))))
        x = x.transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False,
            )
            encoded, _ = self.lstm(packed)
            encoded, _ = nn.utils.rnn.pad_packed_sequence(
                encoded, batch_first=True, total_length=x.shape[1],
            )
        else:
            encoded, _ = self.lstm(x)
        context, _ = self.attention(encoded, mask)
        return self.classifier(context)
