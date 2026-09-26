#!/usr/bin/env python3
"""
Exporta modelo treinado para produção.

Gera artefato leve contendo: model_state_dict, label_map, config, feature_dim.
Opcionalmente exporta para TorchScript.

Uso:
    python scripts/export_model.py --checkpoint experiments/bilstm_48f/best.pt --output artifacts/
"""
import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Exportar modelo para produção")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--label-map", type=str, default="data/processed/label_map.json")
    parser.add_argument("--output", type=str, default="artifacts")
    parser.add_argument("--torchscript", action="store_true", help="Exportar TorchScript")
    parser.add_argument("--input-dim", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carregar checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_class = ckpt.get("model_class", "BiLSTMModel").lower().replace("model", "")
    num_classes = ckpt["num_classes"]

    # Label map
    with open(args.label_map) as f:
        label_map = json.load(f)

    # Artefato de produção
    production_artifact = {
        "model_state_dict": ckpt["model_state_dict"],
        "model_type": model_class,
        "num_classes": num_classes,
        "label_map": label_map,
        "config": ckpt.get("config", {}),
        "val_metrics": ckpt.get("val_metrics", {}),
        "epoch": ckpt.get("epoch", 0),
    }

    artifact_path = output_dir / "sinaliza_model.pt"
    torch.save(production_artifact, artifact_path)
    logger.info(f"Artefato de produção salvo em {artifact_path}")

    # TorchScript (opcional)
    if args.torchscript:
        from ml.models import create_model
        input_dim = args.input_dim
        if input_dim is None:
            logger.error("--input-dim é necessário para TorchScript")
            return

        model = create_model(model_class, input_dim=input_dim, num_classes=num_classes)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

        try:
            dummy_input = torch.randn(1, 48, input_dim)
            scripted = torch.jit.trace(model, dummy_input)
            ts_path = output_dir / "sinaliza_model_scripted.pt"
            scripted.save(str(ts_path))
            logger.info(f"TorchScript salvo em {ts_path}")
        except Exception as e:
            logger.warning(f"Falha no TorchScript: {e}")

    # Metadados JSON
    meta = {
        "model_type": model_class,
        "num_classes": num_classes,
        "source_checkpoint": args.checkpoint,
        "val_metrics": ckpt.get("val_metrics", {}),
    }
    with open(output_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Exportação concluída!")


if __name__ == "__main__":
    main()
