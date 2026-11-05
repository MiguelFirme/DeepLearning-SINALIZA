"""Shape, padding and experiment config checks for the new architecture."""
import torch

from ml.models import create_model
from scripts.train import load_config


def test_masked_padding_does_not_change_logits():
    torch.manual_seed(42)
    model = create_model("conv_bilstm_attention", input_dim=697, num_classes=25,
                         projection_dim=64, conv_channels=64, hidden_size=32,
                         num_layers=2, classifier_hidden=64, dropout=0.45).eval()
    frames = torch.randn(2, 12, 697)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[0, :8] = True
    mask[1, :5] = True
    changed = frames.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 1000
    with torch.inference_mode():
        logits = model(frames, mask)
        repeated = model(changed, mask)
    assert logits.shape == (2, 25)
    torch.testing.assert_close(logits, repeated, atol=1e-6, rtol=0)
    assert torch.isfinite(logits).all()


def test_new_configs_only_change_architecture_and_feature_mode():
    baseline = load_config("configs/transformer_mixed25_pose_face.yaml")
    pose = load_config("configs/conv_bilstm_attention_mixed25_pose_face.yaml")
    full = load_config("configs/conv_bilstm_attention_mixed25_full.yaml")
    for key in ("training", "augmentation", "seed"):
        assert pose.get(key) == baseline.get(key)
        assert full.get(key) == baseline.get(key)
    assert pose["data"] == baseline["data"]
    assert full["data"] == {**baseline["data"], "feature_mode": "full"}
    assert pose["model"]["type"] == "conv_bilstm_attention"
    assert full["model"] == pose["model"]
