"""
Avaliação completa de modelos e geração de relatórios.

Inclui:
    - Avaliação no conjunto de teste
    - Plots: matriz de confusão, loss, accuracy, F1, LR
    - Relatório Markdown de treinamento
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.training.metrics import MetricsCalculator, MetricsResult

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Avaliação completa de modelos treinados."""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        num_classes: int,
        label_names: dict[int, str] | None = None,
        device: str | torch.device | None = None,
        output_dir: str | Path = "reports",
    ):
        self.model = model
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.label_names = label_names or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self) -> MetricsResult:
        """Avalia no conjunto de teste."""
        self.model.eval()
        calc = MetricsCalculator(self.num_classes, self.label_names)

        for batch in self.test_loader:
            seq = batch["sequence"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch["label"]
            logits = self.model(seq, mask).cpu().numpy()
            calc.update(logits, labels.numpy())

        result = calc.compute()
        logger.info(calc.summary_string(result))
        return result

    def plot_confusion_matrix(self, result: MetricsResult, filename: str = "confusion_matrix.png", top_n: int = 30):
        """Plota matriz de confusão (top_n classes mais frequentes)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib não disponível. Pulando plot.")
            return

        cm = result.confusion_matrix
        class_sums = cm.sum(axis=1)
        top_idx = np.argsort(class_sums)[-top_n:][::-1]

        sub_cm = cm[np.ix_(top_idx, top_idx)]
        names = [self.label_names.get(i, str(i))[:15] for i in top_idx]

        fig, ax = plt.subplots(figsize=(max(12, top_n * 0.5), max(10, top_n * 0.4)))
        im = ax.imshow(sub_cm, cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de Confusão (top {top_n} classes)")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Confusion matrix salva em {path}")

    def plot_training_history(self, history_path: str | Path, prefix: str = "history"):
        """Plota curvas de treinamento a partir do JSON de histórico."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib não disponível.")
            return

        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)

        epochs = [h["epoch"] for h in history]

        plots = [
            ("loss", "train_loss", "val_loss", "Loss"),
            ("accuracy", "train_top1", "val_top1", "Top-1 Accuracy"),
            ("f1", None, "val_f1_macro", "F1 Macro"),
            ("lr", "lr", None, "Learning Rate"),
        ]

        for name, train_key, val_key, title in plots:
            fig, ax = plt.subplots(figsize=(10, 5))
            if train_key and train_key in history[0]:
                ax.plot(epochs, [h[train_key] for h in history], label="Train", linewidth=1.5)
            if val_key and val_key in history[0]:
                ax.plot(epochs, [h[val_key] for h in history], label="Val", linewidth=1.5)
            ax.set_xlabel("Época")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path = self.output_dir / f"{prefix}_{name}.png"
            plt.savefig(path, dpi=150)
            plt.close()
            logger.info(f"Plot salvo em {path}")

    def generate_report(
        self,
        result: MetricsResult,
        training_info: dict | None = None,
        model_info: dict | None = None,
        filename: str = "training_report.md",
    ) -> str:
        """Gera relatório Markdown completo."""
        lines = [
            "# Relatório de Treinamento — Sinaliza",
            "",
            "## 1. Informações do Modelo",
        ]

        if model_info:
            for k, v in model_info.items():
                lines.append(f"- **{k}**: {v}")
        lines.append("")

        if training_info:
            lines.append("## 2. Treinamento")
            for k, v in training_info.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        lines.extend([
            "## 3. Métricas de Teste",
            "",
            f"| Métrica | Valor |",
            f"|---------|-------|",
            f"| Top-1 Accuracy | {result.top1_accuracy:.4f} |",
            f"| Top-3 Accuracy | {result.top3_accuracy:.4f} |",
            f"| Top-5 Accuracy | {result.top5_accuracy:.4f} |",
            f"| Precision (macro) | {result.precision_macro:.4f} |",
            f"| Recall (macro) | {result.recall_macro:.4f} |",
            f"| F1 Score (macro) | {result.f1_macro:.4f} |",
            f"| F1 Score (weighted) | {result.f1_weighted:.4f} |",
            f"| Total de amostras | {result.total_samples} |",
            "",
        ])

        if result.most_confused:
            lines.extend([
                "## 4. Pares Mais Confundidos",
                "",
                "| Real | Predito | Ocorrências |",
                "|------|---------|-------------|",
            ])
            for real, pred, count in result.most_confused[:10]:
                lines.append(f"| {real} | {pred} | {count} |")
            lines.append("")

        # Bottom classes
        bottom_classes = sorted(result.per_class_accuracy.items(), key=lambda x: x[1])[:10]
        if bottom_classes:
            lines.extend([
                "## 5. Classes com Menor Acurácia",
                "",
                "| Classe | Acurácia |",
                "|--------|----------|",
            ])
            for cls_id, acc in bottom_classes:
                name = self.label_names.get(cls_id, str(cls_id))
                lines.append(f"| {name} | {acc:.4f} |")
            lines.append("")

        report = "\n".join(lines)
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Relatório salvo em {path}")
        return report
