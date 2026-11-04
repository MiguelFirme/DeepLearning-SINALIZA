"""The archived 16-point fusion must match the offline evaluation on one video."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from backend.services.fusion import FusionBundle


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/fusion25_face16_fixed_rule.json"


@pytest.mark.skipif(not SPEC.exists(), reason="face16 experiment not archived yet")
def test_face16_runtime_matches_offline_probabilities():
    bundle = FusionBundle(SPEC, "cpu")
    assert bundle.feature_modes == {"pose_face": "pose_face16", "full": "full_face16"}
    with (ROOT / "experiments/fusion_mixed25_face16/val_pose_face_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        row = next(item for item in csv.DictReader(handle) if item["top_k"] == "5")
    label_map = json.loads((ROOT / bundle.spec["label_map"]).read_text(encoding="utf-8"))
    names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    with np.load(ROOT / "data/landmarks/mixed_25" / row["path"]) as data:
        actual, _ = bundle.probabilities(data["landmarks"], data["mask"])
    for name, field in (("pose_face", "pose_probabilities"), ("full", "full_probabilities")):
        expected = np.asarray([json.loads(row[field])[label] for label in names])
        np.testing.assert_allclose(actual[name], expected, atol=1e-5)
