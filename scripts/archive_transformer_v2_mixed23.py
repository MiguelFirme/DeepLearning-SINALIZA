#!/usr/bin/env python3
"""Arquiva o checkpoint v2 mixed23 e a evidência de seu treinamento."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import torch


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive():
    experiment = ROOT / "experiments/transformer_v2_mixed23_seed42"
    logs = ROOT / "logs/experiments/transformer_v2_mixed23_seed42"
    processed = ROOT / "data/processed/mixed_23"
    output = ROOT / "Modelos salvos/transformer_v2_mixed23_full"
    record = json.loads((logs / "run_record.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(experiment / "best.pt", map_location="cpu", weights_only=False)
    labels = json.loads((processed / "label_map.json").read_text(encoding="utf-8"))
    if record["status"] != "completed":
        raise ValueError("O treino não foi concluído")
    if (checkpoint.get("pipeline_version") != 2 or checkpoint.get("feature_dim") != 725
            or checkpoint.get("num_classes") != 23 or checkpoint.get("model_type") != "transformer"
            or checkpoint.get("label_map") != labels or checkpoint.get("target_frames") != 48
            or checkpoint.get("target_fps") != 15 or checkpoint.get("window_seconds") != 3.2
            or checkpoint.get("epoch") + 1 != record["result"]["best_epoch"]):
        raise ValueError("Checkpoint diferente do experimento v2 de 23 classes esperado")
    for path, expected in record["hashes"].items():
        current = Path(path)
        if not current.is_absolute():
            current = ROOT / current
        if not current.is_file() or digest(current) != expected:
            raise ValueError(f"Os dados ou índices mudaram após o treino: {current}")
    copies = [(experiment / "best.pt", output / "best.pt")]
    copies += [(ROOT / "configs" / name, output / "config" / name) for name in
               ("transformer_v2_mixed23.yaml", "transformer_v2_mixed25.yaml", "augmentation_v2.yaml")]
    copies += [(processed / name, output / "data_split" / name) for name in
               ("manifest.json", "splits.json", "label_map.json", "dataset_meta.json")]
    copies += [(experiment / name, output / "records" / name) for name in
               ("history.json", "history.csv", "val_metrics.json")]
    copies += [(logs / name, output / "records" / name) for name in ("train.log", "run_record.json")]
    for source, target in copies:
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists() and digest(source) != digest(target):
            raise ValueError(f"Arquivo arquivado diferente; não sobrescrito: {target}")
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    metrics = json.loads((experiment / "val_metrics.json").read_text(encoding="utf-8"))
    readme = f"""# Transformer v2 de 23 classes — entrada completa

Modelo padrão ativado em 13/11/2026. Arquivo de pesos: `best.pt`.

Treino: `transformer_v2_mixed23_seed42`, 440/75/75 vídeos por pessoa em
treino/validação/teste; 13 classes MINDS e dez do grupo. Seleção por F1 macro da
validação, época {record['result']['best_epoch']}; early stopping na época {record['result']['total_epochs']}.
Validação: {metrics['top1']:.1%}, F1 macro {metrics['f1_macro']:.4f}.
O conjunto de teste não foi avaliado nesta ativação.

Pipeline v2: correção de aspecto, pose superior sem visibilidade, mãos globais e
locais, distâncias e velocidade por segundo; 725 features, 15 FPS, janela 3,2 s.
Transformer com dimensão 128, quatro cabeças, três camadas e dropout 0,15.
AdamW 1e-3/L2 0,01, warmup 5, cosine, sampler balanceado e augmentation no treino.
Não há classe `_nenhum_sinal` treinada neste checkpoint.

O checkpoint inclui mapa de classes e configuração de pré-processamento. Não
precisa dos dados brutos para inferência. `config/`, `data_split/` e `records/`
preservam o protocolo, configurações e logs. `checksums.sha256` verifica os arquivos.
Leia `docs/RESULTADO_TREINO_V2_MIXED23_2026-11-12.md` e
`docs/ATIVACAO_TRANSFORMER_V2_PADRAO_2026-11-13.md` para limites e execução.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    checksums = [f"{digest(path)}  {path.relative_to(output).as_posix()}" for path in sorted(output.rglob("*"))
                 if path.is_file() and path.name != "checksums.sha256"]
    (output / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(output), "checkpoint_sha256": digest(output / "best.pt"),
                      "pipeline_version": 2, "classes": len(labels), "best_epoch": record['result']['best_epoch']}, indent=2))


if __name__ == "__main__":
    archive()
