"""Regression checks for the frozen 25-class runtime path."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from backend.services.fusion import FusionBundle
from backend.config import settings
from backend.services.prediction import PredictionService
from scripts.calibrate_fusion25 import apply_temperature, candidate_fusion


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/fusion25_frozen.json"


@pytest.mark.skipif(not MANIFEST.exists(), reason="Frozen artifact unavailable")
def test_runtime_agrees_with_offline_prediction():
    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bundle = FusionBundle(MANIFEST, "cpu")
    with (ROOT / "experiments/fusion_mixed25/val_pose_face_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        row = next(r for r in csv.DictReader(handle) if r["top_k"] == "5")
    sample_path = ROOT / "data/landmarks/mixed_25" / row["path"]
    if not sample_path.is_file():
        pytest.skip("Optional training landmark video is not available locally")
    with np.load(sample_path) as data:
        probabilities, _ = bundle.probabilities(data["landmarks"], data["mask"])
    labels = json.loads((ROOT / spec["label_map"]).read_text(encoding="utf-8"))
    names = [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]
    for mode in ("pose_face", "full"):
        expected = np.array([json.loads(row[f"{mode if mode == 'full' else 'pose'}_probabilities"])[name]
                             for name in names])
        expected = apply_temperature(expected[None], spec["temperatures"][mode])[0]
        np.testing.assert_allclose(probabilities[mode], expected, atol=1e-5)
    scores, candidates = bundle.score(probabilities)
    offline = candidate_fusion(probabilities["pose_face"][None], probabilities["full"][None],
                               spec["fusion"]["first"], spec["fusion"]["top_k"],
                               spec["fusion"]["pose_weight"])[0]
    np.testing.assert_allclose(scores, offline, atol=1e-6)
    assert len(candidates) == 5


def test_rejects_tampered_manifest_hash(tmp_path):
    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    spec["full_sha256"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="Hash diferente"):
        FusionBundle(path, "cpu")


def test_runtime_can_load_without_training_data(tmp_path):
    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name in ("split", "data_manifest", "validation_predictions"):
        spec[f"{name}_path"] = str(tmp_path / f"missing_{name}.json")
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert FusionBundle(path, "cpu").target_frames == 48


def test_backend_session_uses_25_class_fusion(monkeypatch):
    monkeypatch.setattr(settings, "fusion_manifest_path", str(MANIFEST))
    service = PredictionService()
    service.load_model()
    assert service.get_info()["num_classes"] == 25
    session = service.new_session_predictor()
    manifest_path = ROOT / "data/landmarks/mixed_25/manifest.json"
    if not manifest_path.is_file():
        pytest.skip("Optional training landmark manifest is not available locally")
    samples = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_path = ROOT / "data/landmarks/mixed_25" / samples[0]["path"]
    if not sample_path.is_file():
        pytest.skip("Optional training landmark video is not available locally")
    with np.load(sample_path) as data:
        result = service.predict(data["landmarks"], data["mask"], predictor=session)
    assert result["fusion_mode"] == "fusion"
    assert len(result["top_k"]) <= 5
    assert result["processing_time_ms"] > 0
    assert "full_inference_ms" in result["timings"]
