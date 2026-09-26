#!/usr/bin/env python3
"""
Avaliação de modelo treinado no conjunto de teste.

Uso:
    python scripts/evaluate.py --checkpoint experiments/bilstm_48f/best.pt --data-dir data/landmarks
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Avaliar modelo Sinaliza")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data/landmarks")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default="reports")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Carregar checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    model_class = ckpt.get("model_class", "BiLSTMModel").lower().replace("model", "")

    # Carregar label map e splits
    processed_dir = Path(args.processed_dir)
    with open(processed_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    with open(processed_dir / "splits.json", encoding="utf-8") as f:
        splits = json.load(f)

    inv_label = {v: k for k, v in label_map.items()}

    # Dataset de teste
    from ml.data.dataset import LibrasDataset
    data_dir = Path(args.data_dir)
    target_frames = int(ckpt.get("target_frames", 48))
    model_kwargs = dict(ckpt.get("model_kwargs", {}))
    model_type = ckpt.get("model_type") or model_class
    if args.config and not model_kwargs:
        from scripts.train import load_config
        cfg = load_config(args.config)
        target_frames = int(cfg.get("data", {}).get("target_frames", target_frames))
        model_cfg = cfg.get("model", {})
        model_type = model_cfg.get("type", model_type)
        model_kwargs = {
            key: value
            for key, value in model_cfg.items()
            if key not in {"type", "input_dim", "num_classes"}
        }
        if model_type == "bilstm" and "hidden_dim" in model_kwargs:
            model_kwargs["hidden_size"] = model_kwargs.pop("hidden_dim")
        if model_type == "transformer" and "max_len" in model_kwargs:
            model_kwargs["max_seq_len"] = model_kwargs.pop("max_len")

    test_ds = LibrasDataset(
        data_dir,
        label_map=label_map,
        target_frames=target_frames,
        split_indices=splits["test"],
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f"Test set: {len(test_ds)} amostras")

    # Reconstruir modelo
    from ml.models import create_model
    input_dim = test_ds.feature_dim
    model = create_model(model_type, input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Avaliar
    from ml.evaluation.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(model, test_loader, num_classes, inv_label, device, args.output)
    result = evaluator.evaluate()

    # Plots e relatório
    evaluator.plot_confusion_matrix(result)

    history_path = Path(args.checkpoint).parent / "history.json"
    if history_path.exists():
        evaluator.plot_training_history(history_path)

    training_info = {"checkpoint": args.checkpoint, "best_epoch": ckpt.get("epoch", "?")}
    model_info = model.get_model_info() if hasattr(model, "get_model_info") else {}
    evaluator.generate_report(result, training_info, model_info)

    logger.info("Avaliação concluída!")


if __name__ == "__main__":
    main()
