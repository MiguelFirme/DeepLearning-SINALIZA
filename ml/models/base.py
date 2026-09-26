"""Base model class."""
from abc import ABC, abstractmethod
import torch, torch.nn as nn

class BaseSignModel(nn.Module, ABC):
    def __init__(self, input_dim: int, num_classes: int, **kw):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor: ...

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": train}

    def get_model_info(self):
        p = self.count_parameters()
        return {"type": self.__class__.__name__, "input_dim": self.input_dim,
                "num_classes": self.num_classes, **p}
