"""Teste do treinador quando não existe validação independente."""

import torch
from torch.utils.data import DataLoader, Dataset

from ml.training.trainer import Trainer, TrainingConfig


class _TinyDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(7)
        self.sequences = torch.randn(12, 6, 8, generator=generator)
        self.labels = torch.arange(12) % 3

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "sequence": self.sequences[index],
            "mask": torch.ones(6, dtype=torch.bool),
            "label": self.labels[index],
        }


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Linear(8, 3)

    def forward(self, sequence, mask=None):
        return self.classifier(sequence.mean(dim=1))


def test_training_without_validation_uses_fixed_epochs_and_train_loss(tmp_path):
    loader = DataLoader(_TinyDataset(), batch_size=4, shuffle=False)
    config = TrainingConfig(
        epochs=2,
        use_amp=False,
        early_stopping=True,
        scheduler_type="none",
        save_dir=str(tmp_path),
        experiment_name="no_val",
    )
    trainer = Trainer(
        _TinyModel(),
        loader,
        None,
        torch.nn.CrossEntropyLoss(),
        num_classes=3,
        config=config,
        device=torch.device("cpu"),
    )

    result = trainer.train()

    assert result["total_epochs"] == 2
    assert trainer.early_stop is None
    assert "train_top1" in trainer.history[0]
    assert "val_loss" not in trainer.history[0]
    checkpoint = torch.load(tmp_path / "no_val" / "best.pt", weights_only=False)
    assert checkpoint["selection_source"] == "train_loss"
