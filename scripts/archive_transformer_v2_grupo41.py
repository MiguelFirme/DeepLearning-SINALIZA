#!/usr/bin/env python3
"""Arquiva o Transformer v2 ampliado de 41 classes para uso no backend."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import load_config


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    name = "transformer_v2_grupo_expandido_20261117_seed42"
    source, logs = ROOT / "experiments" / name, ROOT / "logs/experiments" / name
    data = ROOT / "data/processed/mixed_grupo_20261117"
    output = ROOT / "Modelos salvos/transformer_v2_grupo41_full"
    record = json.loads((logs / "run_record.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
    labels = json.loads((data / "label_map.json").read_text(encoding="utf-8"))
    if record["status"] != "completed" or checkpoint["epoch"] + 1 != record["result"]["best_epoch"]:
        raise ValueError("Treino incompleto ou checkpoint diferente da época selecionada")
    if (checkpoint.get("model_type") != "transformer" or checkpoint.get("pipeline_version") != 2
            or checkpoint.get("feature_dim") != 725 or checkpoint.get("num_classes") != 41
            or len(labels) != 41 or checkpoint.get("label_map") != labels
            or checkpoint.get("target_frames") != 48 or checkpoint.get("target_fps") != 15
            or checkpoint.get("window_seconds") != 3.2):
        raise ValueError("Metadados incompatíveis com o Transformer v2 ampliado")
    if load_config(ROOT / "configs/transformer_v2_grupo_expandido.yaml") != record["config"]:
        raise ValueError("Configuração atual diverge da registrada no treino")
    for name_path, expected in record["hashes"].items():
        path = Path(name_path)
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Dados ou índices alterados após treino: {path}")
    copies = [(source / "best.pt", output / "best.pt")]
    copies += [(ROOT / "configs" / file, output / "config" / file) for file in
               ("transformer_v2_grupo_expandido.yaml", "transformer_v2_mixed23.yaml", "transformer_v2_mixed25.yaml", "augmentation_v2.yaml")]
    copies += [(data / file, output / "data_split" / file) for file in
               ("manifest.json", "label_map.json", "splits.json", "dataset_meta.json")]
    copies += [(source / file, output / "records" / file) for file in ("history.json", "history.csv", "val_metrics.json")]
    copies += [(logs / file, output / "records" / file) for file in ("train.log", "run_record.json")]
    for origin, target in copies:
        if not origin.is_file():
            raise FileNotFoundError(origin)
        if target.exists() and digest(target) != digest(origin):
            raise ValueError(f"Arquivo arquivado divergente; não sobrescrito: {target}")
    for origin, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(origin, target)
    metrics = json.loads((source / "val_metrics.json").read_text(encoding="utf-8"))
    readme = f"""# Transformer v2 ampliado — 41 classes

Modelo padrão ativado em 17/11/2026. Pesos: `best.pt`.
Experimento: `{name}`. Melhor época: {record['result']['best_epoch']}, por F1 macro
da validação; treino encerrado na época {record['result']['total_epochs']}.

Dados: 13 classes MINDS e 28 do grupo, 715 vídeos, splits por pessoa 529/93/93.
Validação: {metrics['top1']:.2%} (70/93), F1 macro {metrics['f1_macro']:.4f}.
Teste não avaliado nesta ativação. A validação não é acurácia de webcam.

Entrada completa: pose superior, face e mãos; 725 features, 15 FPS, janela 3,2 s.
Transformer 128, quatro cabeças, três camadas, dropout 0,15. AdamW, augmentation
somente no treino, sampler balanceado, early stopping por F1 macro. A configuração
e o mapa completo de classes estão no checkpoint e arquivados neste diretório.
Este checkpoint não contém classe `_nenhum_sinal` treinada.

`config/`, `data_split/`, `records/` e `checksums.sha256` guardam a reprodução.
Não é preciso distribuir vídeos/landmarks para inferência. Consulte
`docs/ATIVACAO_TRANSFORMER_V2_41_CLASSES_2026-11-17.md`.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    lines = [f"{digest(path)}  {path.relative_to(output).as_posix()}" for path in sorted(output.rglob("*"))
             if path.is_file() and path.name != "checksums.sha256"]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(output / "best.pt"), "sha256": digest(output / "best.pt"),
                      "classes": len(labels), "best_epoch": record['result']['best_epoch']}, indent=2))


if __name__ == "__main__":
    main()
