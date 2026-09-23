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
