"""The archived ConvTransformer pair reproduces a video-level offline prediction."""
import csv
import json
from pathlib import Path

import numpy as np

from backend.services.fusion import FusionBundle


ROOT = Path(__file__).resolve().parents[1]


def test_archived_conv_transformer_pair_matches_offline_video():
    bundle = FusionBundle(ROOT / "configs/fusion25_conv_transformer_attention_fixed_rule.json", "cpu")
    assert bundle.feature_modes == {"pose_face": "pose_face", "full": "full"}
    label_map = json.loads((ROOT / bundle.spec["label_map"]).read_text(encoding="utf-8"))
    names = [name for name, _ in sorted(label_map.items(), key=lambda pair: pair[1])]
    with (ROOT / bundle.spec["validation_predictions_path"]).open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    with np.load(ROOT / "data/landmarks/mixed_25" / row["path"]) as data:
        actual, _ = bundle.probabilities(data["landmarks"], data["mask"])
    for model_name, csv_field in (("pose_face", "pose_probabilities"), ("full", "full_probabilities")):
        expected = np.asarray([json.loads(row[csv_field])[name] for name in names])
        np.testing.assert_allclose(actual[model_name], expected, atol=5e-5)
    scores, candidates = bundle.score(actual)
    assert [names[index] for index in candidates] == json.loads(row["candidates"])
    assert names[int(scores.argmax())] == row["fused_prediction"]
    np.testing.assert_allclose(scores.max(), float(row["fused_score"]), atol=1e-5)
