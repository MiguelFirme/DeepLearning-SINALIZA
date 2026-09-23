"""
Métricas de avaliação para classificação de sinais.

Inclui:
    - Top-1, Top-3, Top-5 accuracy
    - Precision, Recall, F1 (macro e weighted)
    - Matriz de confusão
    - Pares mais confundidos
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict

import numpy as np


@dataclass
class MetricsResult:
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion_matrix: np.ndarray
    per_class_accuracy: dict[int, float]
    most_confused: list[tuple]
    total_samples: int


class MetricsCalculator:
    """Calcula métricas detalhadas de classificação."""

    def __init__(self, num_classes: int, label_names: dict[int, str] | None = None):
        self.num_classes = num_classes
        self.label_names = label_names or {}
        self.reset()

    def reset(self):
        self._all_preds: list[np.ndarray] = []
        self._all_labels: list[int] = []
        self._all_logits: list[np.ndarray] = []

    def update(self, logits: np.ndarray, labels: np.ndarray):
        """
        Args:
            logits: (B, C) — scores/logits raw
            labels: (B,) — ground truth
        """
        self._all_logits.append(logits)
        self._all_labels.extend(labels.tolist())

    def compute(self) -> MetricsResult:
        all_logits = np.concatenate(self._all_logits, axis=0)  # (N, C)
        all_labels = np.array(self._all_labels)
        N = len(all_labels)

        # Top-K accuracy
        top1 = self._topk_accuracy(all_logits, all_labels, k=1)
        top3 = self._topk_accuracy(all_logits, all_labels, k=3)
        top5 = self._topk_accuracy(all_logits, all_labels, k=5)

        # Predições para métricas de classe
        preds = all_logits.argmax(axis=1)

        # Confusion matrix
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for pred, true in zip(preds, all_labels):
            cm[true, pred] += 1

        # Per-class metrics
        precision_per = np.zeros(self.num_classes)
        recall_per = np.zeros(self.num_classes)
        f1_per = np.zeros(self.num_classes)
        support = np.zeros(self.num_classes)
        per_class_acc = {}

        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            total_c = cm[c, :].sum()
            support[c] = total_c

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0

            precision_per[c] = p
            recall_per[c] = r
            f1_per[c] = f
            per_class_acc[c] = tp / total_c if total_c > 0 else 0

        # Macro average (ignora classes sem suporte)
        active = support > 0
        prec_macro = precision_per[active].mean() if active.any() else 0
        rec_macro = recall_per[active].mean() if active.any() else 0
        f1_macro = f1_per[active].mean() if active.any() else 0

        # Weighted average
        total_support = support.sum()
        prec_weighted = (precision_per * support).sum() / total_support if total_support > 0 else 0
        rec_weighted = (recall_per * support).sum() / total_support if total_support > 0 else 0
        f1_weighted = (f1_per * support).sum() / total_support if total_support > 0 else 0

        # Most confused pairs
        confused = self._find_most_confused(cm, top_n=10)

        return MetricsResult(
            top1_accuracy=top1,
            top3_accuracy=top3,
            top5_accuracy=top5,
            precision_macro=prec_macro,
            recall_macro=rec_macro,
            f1_macro=f1_macro,
            precision_weighted=prec_weighted,
            recall_weighted=rec_weighted,
            f1_weighted=f1_weighted,
            confusion_matrix=cm,
            per_class_accuracy=per_class_acc,
            most_confused=confused,
            total_samples=N,
        )

    @staticmethod
    def _topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
        k = min(k, logits.shape[1])
        topk_preds = np.argsort(logits, axis=1)[:, -k:]  # top-k indices
        correct = np.any(topk_preds == labels[:, np.newaxis], axis=1)
        return correct.mean().item()

    def _find_most_confused(self, cm: np.ndarray, top_n: int = 10) -> list[tuple]:
        """Encontra os pares (real, previsto) mais confundidos."""
        pairs = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and cm[i, j] > 0:
                    name_i = self.label_names.get(i, str(i))
                    name_j = self.label_names.get(j, str(j))
                    pairs.append((name_i, name_j, int(cm[i, j])))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_n]

    def summary_string(self, result: MetricsResult) -> str:
        lines = [
            f"{'='*50}",
            f"  MÉTRICAS DE AVALIAÇÃO ({result.total_samples} amostras)",
            f"{'='*50}",
            f"  Top-1 Accuracy: {result.top1_accuracy:.4f}",
            f"  Top-3 Accuracy: {result.top3_accuracy:.4f}",
            f"  Top-5 Accuracy: {result.top5_accuracy:.4f}",
            f"  Precision (macro): {result.precision_macro:.4f}",
            f"  Recall (macro):    {result.recall_macro:.4f}",
            f"  F1 (macro):        {result.f1_macro:.4f}",
            f"  F1 (weighted):     {result.f1_weighted:.4f}",
        ]
        if result.most_confused:
            lines.append(f"\n  Pares mais confundidos:")
            for real, pred, count in result.most_confused[:5]:
                lines.append(f"    {real} → {pred}: {count}×")
        return "\n".join(lines)
