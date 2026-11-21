#!/usr/bin/env python3
"""Arquiva o retreino de 39 classes sem modificar pesos nem índices anteriores."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train import load_config


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    name = "transformer_v2_grupo39_20261120_seed42"
    source, logs = ROOT / "experiments" / name, ROOT / "logs/experiments" / name
    processed = ROOT / "data/processed/mixed_grupo39_20261120"
    output = ROOT / "Modelos salvos/transformer_v2_grupo39_full"
    record = json.loads((logs / "run_record.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
    labels = json.loads((processed / "label_map.json").read_text(encoding="utf-8"))
    if record["status"] != "completed" or checkpoint["epoch"] + 1 != record["result"]["best_epoch"]:
        raise ValueError("Treino incompleto ou checkpoint de época divergente")
    if (checkpoint.get("model_type") != "transformer" or checkpoint.get("pipeline_version") != 2
            or checkpoint.get("feature_dim") != 725 or checkpoint.get("num_classes") != 39
            or checkpoint.get("label_map") != labels or len(labels) != 39
            or {"Cinco", "Vacina", "Amarelo"} & set(labels)
            or checkpoint.get("target_fps") != 15 or checkpoint.get("window_seconds") != 3.2):
        raise ValueError("Metadados ou vocabulário incompatíveis")
    if load_config(ROOT / "configs/transformer_v2_grupo39.yaml") != record["config"]:
        raise ValueError("Configuração diferente da registrada no treino")
    for filename, expected in record["hashes"].items():
        path = ROOT / filename
        if digest(path) != expected:
            raise ValueError(f"Índice alterado após treino: {path}")
    copies = [(source / "best.pt", output / "best.pt")]
    copies += [(ROOT / "configs" / name, output / "config" / name) for name in
               ("transformer_v2_grupo39.yaml", "transformer_v2_grupo_expandido.yaml",
                "transformer_v2_mixed23.yaml", "transformer_v2_mixed25.yaml", "augmentation_v2.yaml")]
    copies += [(processed / name, output / "data_split" / name) for name in
               ("manifest.json", "splits.json", "label_map.json", "dataset_meta.json")]
    copies += [(source / name, output / "records" / name) for name in
               ("history.csv", "history.json", "val_metrics.json")]
    copies += [(logs / name, output / "records" / name) for name in ("train.log", "run_record.json")]
    for origin, target in copies:
        if not origin.is_file() or (target.exists() and digest(target) != digest(origin)):
            raise ValueError(f"Origem ausente ou arquivo arquivado divergente: {origin}")
    for origin, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(origin, target)
    metrics = json.loads((source / "val_metrics.json").read_text(encoding="utf-8"))
    summary = f"""# Transformer v2 de 39 classes

Retreino de 20/11/2026 sem Cinco, Vacina e Amarelo. Cinco já estava ausente.
Experimento: `{name}`. Peso treinado: `best.pt`.
635 vídeos, separados por pessoa em 469 treino, 83 validação e 83 teste.
Mesmo Transformer full de 725 features: pose, 40 pontos de face e mãos;
48 amostras a 15 FPS, janela de 3,2 s. d_model 128, 4 cabeças, 3 camadas,
dropout 0,15. AdamW, LR 0,001, weight_decay 0,01, cosine/warmup 5 épocas,
label smoothing 0,1, sampler balanceado e augmentation somente no treino.

Seleção por F1 macro de validação (paciência 25). Melhor época:
{record['result']['best_epoch']}; épocas executadas: {record['result']['total_epochs']}.
Validação: top1 {metrics['top1']:.6%}, F1 macro {metrics['f1_macro']:.6f}.
Teste reservado e não avaliado. Isso não é uma métrica de webcam.
Não há classe negativa `_nenhum_sinal` treinada.

Configurações, índices, mapa de classes, logs e métricas acompanham o modelo.
`checksums.sha256` permite conferir a integridade dos arquivos.
Reprodução e ativação: `docs/RETREINO_GRUPO39_2026-11-20.md` no repositório.
"""
    (output / "README.md").write_text(summary, encoding="utf-8")
    lines = [f"{digest(path)}  {path.relative_to(output).as_posix()}"
             for path in sorted(output.rglob("*")) if path.is_file() and path.name != "checksums.sha256"]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(output / "best.pt"), "sha256": digest(output / "best.pt"),
                      "classes": len(labels), "result": record["result"], "val_top1": metrics["top1"],
                      "val_f1_macro": metrics["f1_macro"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
