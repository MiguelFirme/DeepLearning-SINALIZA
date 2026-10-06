"""Proteções da avaliação por origem e augmentation somente no treino."""
import json

import numpy as np
import pytest
import torch

from ml.data.augmentation import AugmentationConfig, SequenceAugmentor
from ml.data.dataset import LibrasDataset
from ml.data.split import DataSplitter, SplitConfig
from ml.models.bilstm import BiLSTMModel
from ml.training.trainer import Trainer, TrainingConfig
from torch.utils.data import DataLoader


def test_group_shuffle_keeps_source_video_together():
    samples = []
    for number in range(20):
        source = f"video_{number}.npz"
        samples.append({"path": source, "label": str(number % 2), "signer": "A"})
        samples.append({"path": f"video_{number}_syn.npz", "source_path": source,
                        "label": str(number % 2), "signer": "A", "is_synthetic": True})
    splitter = DataSplitter(SplitConfig(strategy="group", group_key="video"))
    splits = splitter.split(samples)
    assert all(splits.values())
    splitter.verify_group_isolation(samples, splits)
    assert sum(map(len, splits.values())) == len(samples)
    broken = {key: list(value) for key, value in splits.items()}
    source_index = broken["train"][0]
    source = splitter.group_id(samples[source_index])
    sibling = next(i for i in broken["train"] if i != source_index and splitter.group_id(samples[i]) == source)
    broken["train"].remove(sibling)
    broken["test"].append(sibling)
    with pytest.raises(ValueError, match="Vazamento de grupo"):
        splitter.verify_group_isolation(samples, broken)


def test_train_augmentation_changes_each_read_and_validation_is_clean(tmp_path):
    landmarks = np.full((8, 346), 0.5, dtype=np.float32)
    mask = np.ones((8, 4), dtype=bool)
    np.savez(tmp_path / "video.npz", landmarks=landmarks, mask=mask)
    (tmp_path / "manifest.json").write_text(json.dumps([
        {"path": "video.npz", "label": "sinal", "signer": "A"}
    ]), encoding="utf-8")
    config = AugmentationConfig(enabled=True, prob=1.0, noise_prob=1.0,
                                noise_std=0.01, scale_enabled=False,
                                translate_enabled=False, rotate_enabled=False,
                                temporal_crop_enabled=False, frame_drop_enabled=False,
                                speed_enabled=False, landmark_mask_enabled=False,
                                temporal_jitter_enabled=False)
    train = LibrasDataset(tmp_path, augmentor=SequenceAugmentor(config, np.random.default_rng(1)))
    val = LibrasDataset(tmp_path)
    assert not torch.equal(train[0]["sequence"], train[0]["sequence"])
    before = train[0]["sequence"]
    train.set_epoch(1)
    assert not torch.equal(before, train[0]["sequence"])
    assert torch.equal(val[0]["sequence"], val[0]["sequence"])
    assert val.augmentor is None


def test_regularized_model_and_training_controls(tmp_path):
    model = BiLSTMModel(346, 3, hidden_size=64, num_layers=1,
                        projection_dim=64, classifier_hidden=64,
                        dropout=0.45, regularized=True)
    assert any(isinstance(layer, torch.nn.LayerNorm) for layer in model.projection)
    assert any(isinstance(layer, torch.nn.LayerNorm) for layer in model.classifier)
    assert all(abs(layer.p - 0.45) < 1e-8 for layer in model.modules()
               if isinstance(layer, torch.nn.Dropout))
    assert model(torch.randn(2, 8, 346), torch.ones(2, 8, dtype=torch.bool)).shape == (2, 3)
    sample = {"sequence": torch.randn(8, 346), "mask": torch.ones(8, dtype=torch.bool),
              "label": torch.tensor(0)}
    loader = DataLoader([sample], batch_size=1)
    config = TrainingConfig(epochs=1, optimizer_type="adam", weight_decay=1e-4, scheduler_type="plateau",
                            early_stopping=True, patience=15, monitor="val_loss",
                            warmup_epochs=0, use_amp=False, save_dir=str(tmp_path))
    trainer = Trainer(model, loader, loader, torch.nn.CrossEntropyLoss(), 3,
                      config=config, device=torch.device("cpu"))
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert trainer.early_stop.patience == 15
    assert trainer.optimizer.param_groups[0]["weight_decay"] == 1e-4
