"""
Modelo B — Transformer 1-D (Pre-Norm) para classificação de sinais.

Arquitetura: LayerNorm → LinearEmbed → Positional Encoding Aprendível
             → TransformerEncoder (pre-norm, N camadas) → CLS Token
             → MLP Head.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from ml.models.base import BaseSignModel


class LearnablePositionalEncoding(nn.Module):
    """Codificação posicional aprendível."""

    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) — já inclui CLS token."""
        return self.dropout(x + self.pos_embed[:, : x.size(1)])


class PreNormTransformerEncoderLayer(nn.Module):
    """Camada do encoder com LayerNorm antes da atenção (pre-norm)."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        # Pre-norm self-attention
        x2 = self.norm1(x)
        x2, _ = self.attn(x2, x2, x2, key_padding_mask=src_key_padding_mask)
        x = x + x2
        # Pre-norm feedforward
        x = x + self.ff(self.norm2(x))
        return x


class TransformerModel(BaseSignModel):
    """Transformer 1-D com CLS token para classificação."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        max_seq_len: int = 128,
        classifier_hidden: int = 256,
        **kwargs,
    ):
        super().__init__(input_dim, num_classes)
        self.d_model = d_model
        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoding = LearnablePositionalEncoding(max_seq_len, d_model, dropout)
        self.encoder_layers = nn.ModuleList([
            PreNormTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
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
        B, T, _ = x.shape
        x = self.input_norm(x)
        x = self.embedding(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)

        # Ajustar máscara para incluir CLS (sempre válido)
        if mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            key_pad_mask = ~torch.cat([cls_mask, mask], dim=1)  # Invert: True=ignore
        else:
            key_pad_mask = None

        x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=key_pad_mask)

        x = self.final_norm(x)
        cls_out = x[:, 0]  # (B, d_model)
        return self.classifier(cls_out)
