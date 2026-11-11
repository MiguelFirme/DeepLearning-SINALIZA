"""Treino sintético curto, checkpoint v2 e compatibilidade com a sessão real."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.config import settings
from backend.services.prediction import PredictionService
from backend.services.video_stream_v2 import TemporalWindowSessionV2
from ml.features.pipeline_v2 import PipelineV2Config
from ml.training.pipeline_v2 import balanced_sampler, run_training_v2
from ml.training.signer_cv_v2 import make_person_folds, run_cv_v2
from scripts.train import load_config
from tests.test_pipeline_v2 import amostra


def test_sampler_por_classe_e_folds():
    sampler = balanced_sampler([0] * 30 + [1] * 5, 42)
    assert sampler.weights[:30].sum() == pytest.approx(sampler.weights[30:].sum())
    manifest = [{"path": str(i), "signer": f"P{i}", "label": "A"} for i in range(4)]
    for person, splits in make_person_folds(manifest):
        assert all(manifest[i]["signer"] == person for i in splits["test"])
        assert not set(splits["train"]) & set(splits["val"])


def test_cv_v2_prepara_folds_sem_annotations(tmp_path):
    manifest = [{"path": f"{i}.npz", "signer": f"P{i}", "label": "A"} for i in range(3)]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    args = SimpleNamespace(data_dir=str(tmp_path), essential_categories=[], extra_keep=[], class_list=None,
                           max_classes=None, run_name="cv", run_root=str(tmp_path / "runs"),
                           processed_root=str(tmp_path / "processed"), signers=None, round="all",
                           dry_run=False, prepare_only=True, seed=42)
    run_cv_v2(args)
    folds = list((tmp_path / "processed/cv").glob("*/splits.json"))
    assert len(folds) == 3
    for path in folds:
        split = json.loads(path.read_text())
        assert len(split["train"]) == len(split["val"]) == len(split["test"]) == 1


def test_treino_checkpoint_carregamento_e_negativo(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    raw, mask, times = amostra(100, 30)
    rows = []
    for person in range(3):
        for label in ("A", "_nenhum_sinal"):
            name = f"{person}_{label}.npz"
            np.savez(tmp_path / name, landmarks=raw, mask=mask,
                     metadata=json.dumps({"fps": 30, "width": 640, "height": 480}))
            rows.append({"path": name, "signer": f"P{person}", "label": label, "origin": "novas"})
    for filename, value in (("manifest.json", rows), ("label_map.json", {"A": 0, "_nenhum_sinal": 1}),
                            ("splits.json", {"train": [0, 1], "val": [2, 3], "test": [4, 5]})):
        (tmp_path / filename).write_text(json.dumps(value), encoding="utf-8")
    cfg = load_config("configs/transformer_v2_mixed25.yaml")
    cfg.pop("augmentation_file")
    cfg["model"]["params"].update(d_model=16, num_layers=1, dim_feedforward=32, classifier_hidden=16)
    cfg["training"].update(epochs=1, use_amp=False)
    args = SimpleNamespace(seed=42, device="cpu", target_frames=None, data_dir=str(tmp_path),
                           processed_dir=str(tmp_path), config="configs/transformer_v2_mixed25.yaml",
                           num_workers=0, batch_size=2, model=None, experiment="sintetico",
                           save_dir=str(tmp_path / "experiments"), log_dir=str(tmp_path / "logs"),
                           lr=None, epochs=1, label_smoothing=None, skip_test=False, test_checkpoint="best")
    run_training_v2(args, cfg)
    path = tmp_path / "experiments/sintetico/best.pt"
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["pipeline_version"] == 2 and checkpoint["feature_dim"] == 725
    assert checkpoint["config"]["monitor"] == "val_f1"
    metrics = json.loads((path.parent / "test_metrics.json").read_text())
    assert metrics["by_origin"]["novas"]["total_samples"] == 2
    assert len(metrics["confusion_matrix"]) == 2
    # Peso controlado para testar o contrato negativo, não uma métrica de modelo.
    checkpoint["model_state_dict"]["classifier.3.weight"].zero_()
    checkpoint["model_state_dict"]["classifier.3.bias"] = torch.tensor([-10., 10.])
    torch.save(checkpoint, path)
    monkeypatch.setattr(settings, "smoothing_type", "none")
    monkeypatch.setattr(settings, "fusion_manifest_path", "")
    monkeypatch.setattr(settings, "model_path", str(path))
    service = PredictionService()
    service.load_model()
    assert service.pipeline_version == 2
    result = service.predict(raw, mask, timestamps=times, width=640, height=480)
    assert result["status"] == "no_sign" and result["sign"] is None
    session = TemporalWindowSessionV2(service)
    outputs = []
    for i in range(51):
        reply = session.process_landmarks(raw[0], mask[0], i / 15, 640, 480)
        if reply:
            outputs.append(reply)
    assert outputs and outputs[0]["status"] == "no_sign"
