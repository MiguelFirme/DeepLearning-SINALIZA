"""Testes para o pipeline de normalização de landmarks."""
import numpy as np
import pytest

from ml.features.normalization import LandmarkNormalizer, NormalizationConfig


@pytest.fixture
def normalizer():
    return LandmarkNormalizer(NormalizationConfig())


@pytest.fixture
def dummy_sequence():
    """Sequência dummy: (T=10, D=346)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 346)).astype(np.float32)


def test_normalizer_output_shape(normalizer, dummy_sequence):
    """Saída deve ter mesma seq len e dimensão expandida com features extras."""
    out = normalizer.normalize(dummy_sequence)
    assert out.ndim == 2
    assert out.shape[0] == dummy_sequence.shape[0]
    # Dimensão >= original (pode ter velocidade, aceleração, distâncias)
    assert out.shape[1] >= dummy_sequence.shape[1]


def test_normalizer_no_nan(normalizer, dummy_sequence):
    """Resultado não deve conter NaN."""
    out = normalizer.normalize(dummy_sequence)
    assert not np.isnan(out).any()


def test_normalizer_no_inf(normalizer, dummy_sequence):
    """Resultado não deve conter infinitos."""
    out = normalizer.normalize(dummy_sequence)
    assert not np.isinf(out).any()


def test_normalizer_zeros_input(normalizer):
    """Entrada toda zero não deve causar exceção."""
    zeros = np.zeros((10, 346), dtype=np.float32)
    out = normalizer.normalize(zeros)
    assert out.shape[0] == 10
    assert not np.isnan(out).any()


def test_normalizer_single_frame(normalizer):
    """Frame único não deve causar exceção."""
    single = np.random.randn(1, 346).astype(np.float32)
    out = normalizer.normalize(single)
    assert out.shape[0] == 1


def test_missing_parts_remain_zero_in_raw_and_velocity(normalizer, dummy_sequence):
    mask = np.ones((dummy_sequence.shape[0], 4), dtype=bool)
    mask[3, 0] = False
    dummy_sequence[3, :63] = 0.0
    out = normalizer.normalize(dummy_sequence, mask)
    assert np.allclose(out[3, :63], 0.0)
    assert np.allclose(out[3, 346:409], 0.0)


def test_all_five_distances_are_computed_when_parts_are_valid(normalizer, dummy_sequence):
    mask = np.ones((dummy_sequence.shape[0], 4), dtype=bool)
    out = normalizer.normalize(dummy_sequence, mask)
    distances = out[:, -5:]
    assert distances.shape == (dummy_sequence.shape[0], 5)
    assert np.isfinite(distances).all()
    assert np.any(distances[:, 0] > 0)
    assert np.any(distances[:, 4] > 0)
