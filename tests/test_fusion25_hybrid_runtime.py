"""The mixed face16/full40 manifest must reproduce its offline predictions."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from backend.services.fusion import FusionBundle


ROOT = Path(__file__).resolve().parents[1]


def test_hybrid_runtime_matches_offline_predictions():
    bundle = FusionBundle(ROOT / "configs/fusion25_hybrid_face16_full40.json", "cpu")
    assert bundle.feature_modes == {"pose_face": "pose_face16", "full": "full"}
    label_map = json.loads((ROOT / bundle.spec["label_map"]).read_text(encoding="utf-8"))
    names = [name for name, _ in sorted(label_map.items(), key=lambda pair: pair[1])]
    with (ROOT / bundle.spec["validation_predictions_path"]).open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    sample_path = ROOT / "data/landmarks/mixed_25" / row["path"]
    if not sample_path.is_file():
        pytest.skip("Optional training landmark video is not available locally")
    with np.load(sample_path) as data:
        actual, _ = bundle.probabilities(data["landmarks"], data["mask"])
    for model_name, csv_field in (("pose_face", "pose_probabilities"), ("full", "full_probabilities")):
        expected = np.asarray([json.loads(row[csv_field])[name] for name in names])
        np.testing.assert_allclose(actual[model_name], expected, atol=1e-5)
    scores, candidates = bundle.score(actual)
    assert [names[index] for index in candidates] == json.loads(row["candidates"])
    assert names[int(scores.argmax())] == row["fused_prediction"]
    np.testing.assert_allclose(scores.max(), float(row["fused_score"]), atol=1e-5)
