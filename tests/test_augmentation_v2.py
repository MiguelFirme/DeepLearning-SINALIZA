"""Espelhamento e sincronia dos aumentos em dados brutos."""
import numpy as np

from ml.data.augmentation_v2 import RawAugmentorV2, mirror_raw_v2
from ml.features.landmarks import FACE_SELECTED_INDICES
from tests.test_pipeline_v2 import amostra


def test_espelhamento_duplo_e_lateralidade():
    raw, mask, _ = amostra()
    raw += np.random.default_rng(7).normal(0, .01, raw.shape)
    once, once_mask = mirror_raw_v2(raw, mask)
    twice, twice_mask = mirror_raw_v2(once, once_mask)
    valid_columns = np.ones(346, bool)
    index = FACE_SELECTED_INDICES.index(267)
    valid_columns[226 + 3 * index:229 + 3 * index] = False
    np.testing.assert_allclose(twice[:, valid_columns], raw[:, valid_columns], atol=1e-7)
    np.testing.assert_array_equal(twice_mask, mask)
    assert not once[:, ~valid_columns].any()
    np.testing.assert_allclose(once[:, 0], 1 - raw[:, 63])
    mask[:, 0] = False
    raw[:, :63] = 0
    once, once_mask = mirror_raw_v2(raw, mask)
    assert not once[:, 63:126].any() and not once_mask[:, 1].any()


def test_augmentation_sincroniza_mascara_e_relogio():
    raw, mask, times = amostra(150)
    mask[:, 0] = False
    raw[:, :63] = 0
    for seed in range(20):
        out, valid, clock = RawAugmentorV2()(raw, mask, times, np.random.default_rng(seed))
        assert len(out) == len(valid) == len(clock)
        assert len(out) >= 105 and np.all(np.diff(clock) > 0)
        for side in (0, 1):
            assert not out[~valid[:, side], side * 63:(side + 1) * 63].any()
