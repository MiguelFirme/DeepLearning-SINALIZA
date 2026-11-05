"""Architecture and config parity checks for Conv1D + Transformer + attention."""
import torch

from ml.models import create_model
from scripts.train import load_config


def test_masked_padding_does_not_change_predictions():
    torch.manual_seed(42)
    model = create_model("conv_transformer_attention", input_dim=697, num_classes=25,
                         d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                         max_seq_len=48, classifier_hidden=64, conv_channels=64,
                         conv_kernel_size=3, dropout=0.45).eval()
    frames = torch.randn(2, 12, 697)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[0, :8] = True
    mask[1, :5] = True
    changed = frames.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 1000
    with torch.inference_mode():
        logits = model(frames, mask)
        repeated = model(changed, mask)
        empty = model(torch.zeros(1, 12, 697), torch.zeros(1, 12, dtype=torch.bool))
    assert logits.shape == (2, 25)
    torch.testing.assert_close(logits, repeated, atol=1e-6, rtol=0)
    assert torch.isfinite(logits).all() and torch.isfinite(empty).all()


def test_training_configs_match_transformer_baseline():
    baseline = load_config("configs/transformer_mixed25_pose_face.yaml")
    pose = load_config("configs/conv_transformer_attention_mixed25_pose_face.yaml")
    full = load_config("configs/conv_transformer_attention_mixed25_full.yaml")
    for key in ("training", "augmentation", "seed"):
        assert pose.get(key) == baseline.get(key)
        assert full.get(key) == baseline.get(key)
    assert pose["data"] == baseline["data"]
    assert full["data"] == {**baseline["data"], "feature_mode": "full"}
    assert pose["model"]["type"] == "conv_transformer_attention"
    assert full["model"] == pose["model"]
