"""
Modelo A — Bi-LSTM com Atenção Temporal.

Arquitetura: LayerNorm → Projeção Linear → Bi-LSTM (N camadas)
             → Atenção Temporal → Dropout → Classificador MLP.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from ml.models.base import BaseSignModel


class TemporalAttention(nn.Module):
    """Mecanismo de atenção temporal (Bahdanau-style)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, lstm_out: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            lstm_out: (B, T, H)
            mask: (B, T) — True = válido
        Returns:
            context: (B, H)
            weights: (B, T)
        """
        scores = self.attn(lstm_out).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, H)
        return context, weights


class BiLSTMModel(BaseSignModel):
    """Bi-LSTM com atenção temporal para classificação de sinais."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        projection_dim: int = 128,
        classifier_hidden: int = 256,
        **kwargs,
    ):
        super().__init__(input_dim, num_classes)
        self.norm = nn.LayerNorm(input_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * 2
        self.attention = TemporalAttention(lstm_out_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(classifier_hidden, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            mask: (B, T) bool — True = frame válido
        Returns:
            logits: (B, num_classes)
        """
        x = self.norm(x)
        x = self.projection(x)

        if mask is not None:
            lengths = mask.sum(dim=1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)

        context, _ = self.attention(lstm_out, mask)
        return self.classifier(context)
