"""Temporal Conv1D + Transformer encoder + masked attention pooling."""
from __future__ import annotations

import torch
from torch import nn

from ml.models.base import BaseSignModel
from ml.models.bilstm import TemporalAttention
from ml.models.transformer import PreNormTransformerEncoderLayer


class ConvTransformerAttentionModel(BaseSignModel):
    """Recognize isolated signs from a sequence of MediaPipe features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.45,
        max_seq_len: int = 48,
        classifier_hidden: int = 64,
        conv_channels: int = 64,
        conv_kernel_size: int = 3,
    ):
        super().__init__(input_dim, num_classes)
        if conv_kernel_size < 1 or conv_kernel_size % 2 != 1:
            raise ValueError("conv_kernel_size must be a positive odd number")
        if conv_channels != d_model:
            raise ValueError("conv_channels must equal d_model for the encoder")
        if d_model % nhead != 0 or num_layers < 1 or max_seq_len < 1:
            raise ValueError("Invalid Transformer dimensions")
        self.max_seq_len = max_seq_len
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        self.conv = nn.Conv1d(d_model, conv_channels, conv_kernel_size,
                              padding=conv_kernel_size // 2, bias=False)
        self.conv_norm = nn.BatchNorm1d(conv_channels)
        self.conv_activation = nn.GELU()
        self.conv_dropout = nn.Dropout(dropout)
        self.position_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.position_dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            PreNormTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.attention = TemporalAttention(d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != self.input_dim:
            raise ValueError(f"Expected (batch, frames, {self.input_dim}), got {tuple(x.shape)}")
        if x.shape[1] > self.max_seq_len:
            raise ValueError(f"Expected at most {self.max_seq_len} frames")
        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError("mask must have shape (batch, frames)")
            mask = mask.bool().clone()
            mask[~mask.any(dim=1), 0] = True

        x = self.projection(self.input_norm(x))
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = x.transpose(1, 2)
        x = self.conv_dropout(self.conv_activation(self.conv_norm(self.conv(x))))
        x = x.transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = self.position_dropout(x + self.position_embedding[:, :x.shape[1]])
        padding = ~mask if mask is not None else None
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=padding)
        x = self.final_norm(x)
        context, _ = self.attention(x, mask)
        return self.classifier(context)
