"""Dataset v2: metadados auditáveis e janelas em segundos, sem reextração."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict

import numpy as np
import torch

from ml.data.dataset import LibrasDataset
from ml.data.quality import active_interval
from ml.features.pipeline_v2 import PipelineV2Config, FEATURE_DIM_V2, prepare_features_v2

logger = logging.getLogger(__name__)


def read_metadata(data):
    """Aceita dicionário escalar legado ou JSON; não modifica o arquivo."""
    if "metadata" not in data:
        return {}
    value = data["metadata"].item()
    if isinstance(value, (str, bytes)):
        value = json.loads(value)
    return value if isinstance(value, dict) else {}


def window_end(times, start, end, seconds, training, rng):
    """Janela contém >=80% do trecho estimado, ou é sorteada dentro do trecho."""
    a, b = float(times[start]), float(times[max(start, end - 1)])
    duration = b - a
    if not training:
        return (a + b + seconds) / 2
    if duration > seconds:
        return float(rng.uniform(a + seconds, b))
    # Intervalo de inícios com interseção >= 80% da duração ativa.
    low, high = a - seconds + .8 * duration, b - .8 * duration
    return float(rng.uniform(low, high)) + seconds


class LibrasDatasetV2(LibrasDataset):
    """Apenas treino sorteia janelas e aplica aumentos aos dados brutos."""

    def __init__(self, data_dir, label_map=None, split_indices=None, cfg=None,
                 training=False, augmentor=None, augmentation_seed=42, **kwargs):
        super().__init__(data_dir, label_map=label_map, split_indices=split_indices,
                         normalize=False, augmentor=None, augmentation_seed=augmentation_seed)
        self.pipeline_cfg = cfg or PipelineV2Config()
        self.target_frames = self.pipeline_cfg.n_samples
        self.training = training
        if augmentor is not None and not training:
            raise ValueError("Augmentation v2 é permitida somente no treino")
        self.augmentor = augmentor
        self.rng = np.random.default_rng(augmentation_seed)
        self.metadata = []
        self.metadata_issues = {"fps": 0, "dimensions": 0}
        for sample in self.samples:
            with np.load(self.data_dir / sample["path"], allow_pickle=True) as data:
                meta = read_metadata(data)
            try:
                fps = float(meta.get("fps", 0))
            except (ValueError, TypeError):
                fps = 0
            if not np.isfinite(fps) or not 10 <= fps <= 60:
                self.metadata_issues["fps"] += 1
                logger.warning("%s: FPS ausente/inválido; assumindo 30", sample["path"])
                fps = 30.0
            try:
                width, height = float(meta.get("width", 0)), float(meta.get("height", 0))
            except (ValueError, TypeError):
                width, height = 0, 0
            if not np.isfinite([width, height]).all() or min(width, height) <= 0:
                self.metadata_issues["dimensions"] += 1
                logger.warning("%s: dimensões ausentes; aspecto desconhecido, assumindo 1:1", sample["path"])
                width, height = 1., 1.
            self.metadata.append({**meta, "fps": fps, "width": width, "height": height})
        logger.info("Metadados v2, arquivos afetados: %s", self.metadata_issues)

    @property
    def feature_dim(self):
        return FEATURE_DIM_V2

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.rng = np.random.default_rng(self.augmentation_seed + epoch * 100003)

    def __getitem__(self, idx):
        sample, meta = self.samples[idx], self.metadata[idx]
        with np.load(self.data_dir / sample["path"], allow_pickle=True) as data:
            raw = data["landmarks"].astype(np.float32)
            mask = data["mask"].astype(bool)
            times = (data["timestamps"].astype(np.float64) if "timestamps" in data
                     else np.arange(len(raw), dtype=np.float64) / meta["fps"])
        if self.training and self.augmentor is not None:
            raw, mask, times = self.augmentor(raw, mask, times, self.rng)
        start, end = active_interval(raw, mask)
        end_time = window_end(times, start, end, self.pipeline_cfg.window_seconds, self.training, self.rng)
        features, valid, _ = prepare_features_v2(raw, mask, times, meta["width"], meta["height"],
                                               self.pipeline_cfg, end_time=end_time)
        origin = sample.get("origin") or ("novas" if "novas_" in sample["path"] else "minds")
        return {"sequence": torch.from_numpy(features), "mask": torch.from_numpy(valid),
                "label": torch.tensor(self.label_map[sample["label"]], dtype=torch.long),
                "path": sample["path"], "sign_name": sample["label"], "origin": origin}

    def get_stats(self):
        return {**super().get_stats(), "pipeline_version": 2,
                "pipeline_config": asdict(self.pipeline_cfg), "metadata_issues": self.metadata_issues}
