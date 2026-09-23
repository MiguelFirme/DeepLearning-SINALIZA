"""
Dataset PyTorch para landmarks de Libras pré-extraídos (.npz).

Cada arquivo .npz contém:
    - landmarks: (T, D) float32
    - mask: (T, n_parts) bool
    - metadata: dict com fps, total_frames, width, height
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.features.normalization import LandmarkNormalizer, NormalizationConfig
from ml.features.sequence import SequenceProcessor, SequenceConfig

logger = logging.getLogger(__name__)


class LibrasDataset(Dataset):
    """Dataset de landmarks de Libras cached em .npz."""

    def __init__(
        self,
        data_dir: str | Path,
        label_map: dict[str, int] | None = None,
        target_frames: int = 48,
        normalize: bool = True,
        norm_config: NormalizationConfig | None = None,
        seq_config: SequenceConfig | None = None,
        augmentor=None,
        split_indices: list[int] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.target_frames = target_frames
        self.normalize = normalize
        self.augmentor = augmentor

        # Carregar manifesto ou escanear diretório
        self.samples: list[dict] = []
        self._load_samples(split_indices)

        # Mapa de labels
        if label_map is not None:
            self.label_map = label_map
        else:
            self.label_map = self._build_label_map()

        self.num_classes = len(self.label_map)

        # Processadores
        self.seq_processor = SequenceProcessor(seq_config or SequenceConfig(target_frames=target_frames))
        self._normalizer: LandmarkNormalizer | None = None
        self._norm_config = norm_config
        self._feature_dim: int | None = None

    def _load_samples(self, split_indices: list[int] | None = None):
        """Carrega lista de amostras do manifesto ou escaneia o diretório."""
        manifest = self.data_dir / "manifest.json"
        if manifest.exists():
            with open(manifest, encoding="utf-8") as f:
                all_samples = json.load(f)
        else:
            all_samples = []
            for npz_path in sorted(self.data_dir.rglob("*.npz")):
                label = npz_path.parent.name
                all_samples.append({
                    "path": str(npz_path.relative_to(self.data_dir)),
                    "label": label,
                    "signer": "unknown",
                })
        if split_indices is not None:
            self.samples = [all_samples[i] for i in split_indices if i < len(all_samples)]
        else:
            self.samples = all_samples

    def _build_label_map(self) -> dict[str, int]:
        labels = sorted(set(s["label"] for s in self.samples))
        return {label: idx for idx, label in enumerate(labels)}

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is not None:
            return self._feature_dim
        if len(self.samples) > 0:
            data = np.load(str(self.data_dir / self.samples[0]["path"]))
            raw_dim = data["landmarks"].shape[1]
            if self.normalize:
                norm = LandmarkNormalizer(raw_dim, self._norm_config)
                self._feature_dim = norm.get_output_dim()
            else:
                self._feature_dim = raw_dim
            return self._feature_dim
        return 346  # fallback

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        npz_path = self.data_dir / sample["path"]
        data = np.load(str(npz_path), allow_pickle=True)
        landmarks = data["landmarks"].astype(np.float32)  # (T, D)

        # Normalizar
        if self.normalize:
            if self._normalizer is None:
                self._normalizer = LandmarkNormalizer(landmarks.shape[1], self._norm_config)
            landmarks = self._normalizer.normalize_sequence(landmarks)

        # Augmentation
        if self.augmentor is not None:
            landmarks = self.augmentor.augment(landmarks)

        # Processar sequência temporal
        seq, mask = self.seq_processor.process(landmarks)

        label = self.label_map.get(sample["label"], -1)

        return {
            "sequence": torch.from_numpy(seq),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label, dtype=torch.long),
            "path": sample["path"],
            "sign_name": sample["label"],
        }

    def get_labels(self) -> list[int]:
        """Retorna lista de labels (int) para todas as amostras."""
        return [self.label_map.get(s["label"], -1) for s in self.samples]

    def get_class_distribution(self) -> dict[str, int]:
        """Distribuição de classes."""
        dist: dict[str, int] = {}
        for s in self.samples:
            dist[s["label"]] = dist.get(s["label"], 0) + 1
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))

    def get_inverse_label_map(self) -> dict[int, str]:
        return {v: k for k, v in self.label_map.items()}

    def get_stats(self) -> dict:
        return {
            "total_samples": len(self.samples),
            "num_classes": self.num_classes,
            "target_frames": self.target_frames,
            "feature_dim": self.feature_dim,
            "normalize": self.normalize,
        }
