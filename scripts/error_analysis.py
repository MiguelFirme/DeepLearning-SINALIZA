#!/usr/bin/env python3
"""
Análise de erros detalhada: quais classes/amostras o modelo erra e por quê.

Uso:
    python scripts/error_analysis.py --checkpoint experiments/bilstm_48f/best.pt
"""
import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Análise de erros")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/landmarks")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="reports/error_analysis.md")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    model_type = ckpt.get("model_class", "BiLSTMModel").lower().replace("model", "")

    processed_dir = Path(args.processed_dir)
    with open(processed_dir / "label_map.json") as f:
        label_map = json.load(f)
    with open(processed_dir / "splits.json") as f:
        splits = json.load(f)

    inv_label = {v: k for k, v in label_map.items()}

    from ml.data.dataset import LibrasDataset
    from ml.models import create_model

    data_dir = Path(args.data_dir)
    test_ds = LibrasDataset(data_dir, label_map=label_map, split_indices=splits["test"])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    input_dim = test_ds.feature_dim
    model = create_model(model_type, input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()

    # Coletar predições
    all_errors = []
    correct_count = defaultdict(int)
    total_count = defaultdict(int)

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].numpy()
            paths = batch["path"]
            names = batch["sign_name"]

            logits = model(seq, mask).cpu().numpy()
            preds = logits.argmax(axis=1)
            confidences = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

            for i in range(len(labels)):
                true_label = labels[i]
                pred_label = preds[i]
                true_name = names[i]
                total_count[true_name] += 1

                if pred_label == true_label:
                    correct_count[true_name] += 1
                else:
                    pred_name = inv_label.get(pred_label, str(pred_label))
                    conf = confidences[i, pred_label]
                    true_conf = confidences[i, true_label]
                    all_errors.append({
                        "true": true_name, "predicted": pred_name,
                        "confidence": float(conf), "true_confidence": float(true_conf),
                        "path": paths[i],
                    })

    # Gerar relatório
    lines = [
        "# Análise de Erros — Sinaliza",
        "",
        f"Total de erros: {len(all_errors)} / {sum(total_count.values())}",
        "",
        "## Classes com mais erros",
        "",
        "| Classe | Acertos | Total | Acurácia |",
        "|--------|---------|-------|----------|",
    ]

    class_acc = {}
    for name in sorted(total_count.keys()):
        c = correct_count.get(name, 0)
        t = total_count[name]
        acc = c / t if t > 0 else 0
        class_acc[name] = acc

    worst = sorted(class_acc.items(), key=lambda x: x[1])[:20]
    for name, acc in worst:
        c = correct_count.get(name, 0)
        t = total_count[name]
        lines.append(f"| {name} | {c} | {t} | {acc:.2%} |")

    lines.extend(["", "## Erros de alta confiança (modelo 'certo' do erro)", ""])
    high_conf = sorted(all_errors, key=lambda x: x["confidence"], reverse=True)[:15]
    for err in high_conf:
        lines.append(f"- **{err['true']}** → {err['predicted']} (conf={err['confidence']:.3f}, true_conf={err['true_confidence']:.3f})")

    report = "\n".join(lines)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info(f"Relatório salvo em {output_path}")


if __name__ == "__main__":
    main()
