"""Testes da aumentação sincronizada com máscaras de detecção."""

import numpy as np

from ml.data.augmentation import AugmentationConfig, SequenceAugmentor
from scripts.train import load_config


def test_disabled_augmentation_preserves_sequence_and_mask():
    sequence = np.random.default_rng(1).normal(size=(12, 346)).astype(np.float32)
    mask = np.ones((12, 4), dtype=bool)
    augmentor = SequenceAugmentor(AugmentationConfig(enabled=False))
    result, result_mask = augmentor.augment_with_mask(sequence, mask)
    assert np.array_equal(result, sequence)
    assert np.array_equal(result_mask, mask)


def test_missing_hand_remains_zero_after_spatial_augmentation():
    sequence = np.ones((12, 346), dtype=np.float32)
    mask = np.ones((12, 4), dtype=bool)
    mask[4, 0] = False
    sequence[4, :63] = 0.0
    config = AugmentationConfig(
        enabled=True,
        prob=1.0,
        temporal_crop_enabled=False,
        frame_drop_enabled=False,
        speed_enabled=False,
        temporal_jitter_enabled=False,
    )
    augmentor = SequenceAugmentor(config, rng=np.random.default_rng(2))
    result, result_mask = augmentor.augment_with_mask(sequence, mask)
    assert result.shape[0] == result_mask.shape[0]
    assert np.allclose(result[4, :63], 0.0)


def test_temporal_transforms_keep_mask_length_synchronized():
    sequence = np.random.default_rng(3).normal(size=(30, 346)).astype(np.float32)
    mask = np.ones((30, 4), dtype=bool)
    config = AugmentationConfig(enabled=True, prob=1.0)
    augmentor = SequenceAugmentor(config, rng=np.random.default_rng(4))
    result, result_mask = augmentor.augment_with_mask(sequence, mask)
    assert result.shape[0] == result_mask.shape[0]
    assert result_mask.shape[1] == 4


def test_augmented_experiment_is_reproducible_and_preserves_source():
    mapping = load_config("configs/bilstm_augmented.yaml")["augmentation"]
    config = AugmentationConfig.from_mapping(mapping)
    sequence = np.random.default_rng(6).normal(size=(40, 346)).astype(np.float32)
    original = sequence.copy()
    mask = np.ones((40, 4), dtype=bool)
    one = SequenceAugmentor(config, rng=np.random.default_rng(42))
    two = SequenceAugmentor(config, rng=np.random.default_rng(42))
    result_one, mask_one = one.augment_with_mask(sequence, mask)
    result_two, mask_two = two.augment_with_mask(sequence, mask)
    assert np.array_equal(result_one, result_two)
    assert np.array_equal(mask_one, mask_two)
    assert np.array_equal(sequence, original)
    assert not np.array_equal(result_one, sequence)
    assert config.mirror_enabled is False
    assert not config.frame_drop_enabled
    assert not config.landmark_mask_enabled
