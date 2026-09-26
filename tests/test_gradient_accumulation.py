"""A última janela de acumulação deve receber peso integral."""
import copy

import torch
from torch.utils.data import DataLoader, Dataset

from ml.training.trainer import Trainer, TrainingConfig


class TinySequenceDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {
            "sequence": torch.tensor([[float(index), 1.0]]),
            "mask": torch.ones(1, dtype=torch.bool),
            "label": torch.tensor(index % 2),
        }


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, sequence, mask=None):
        return self.linear(sequence[:, 0])


def test_accumulation_remainder_matches_two_full_optimizer_steps(tmp_path):
    torch.manual_seed(9)
    loader = DataLoader(TinySequenceDataset(), batch_size=2, shuffle=False)
    model = TinyClassifier()
    expected_model = copy.deepcopy(model)
    criterion = torch.nn.CrossEntropyLoss()
    config = TrainingConfig(
        epochs=1, learning_rate=1e-3, weight_decay=0.0,
        gradient_accumulation_steps=2, gradient_clip_val=100.0,
        scheduler_type="none", use_amp=False, save_dir=str(tmp_path),
    )
    trainer = Trainer(model, loader, None, criterion, 2, config=config, device=torch.device("cpu"))
    trainer._train_epoch(0)

    optimizer = torch.optim.AdamW(expected_model.parameters(), lr=1e-3, weight_decay=0.0)
    batches = list(loader)
    for window in (batches[:2], batches[2:]):
        optimizer.zero_grad()
        losses = [criterion(expected_model(batch["sequence"]), batch["label"])
                  for batch in window]
        (sum(losses) / len(losses)).backward()
        optimizer.step()

    for actual, expected in zip(model.parameters(), expected_model.parameters()):
        assert torch.allclose(actual, expected, atol=1e-7)
