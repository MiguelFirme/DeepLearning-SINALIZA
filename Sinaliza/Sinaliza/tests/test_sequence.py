"""Testes para o processador de sequências."""
import numpy as np
import pytest

from ml.features.sequence import SequenceProcessor, SequenceConfig


@pytest.fixture
def processor():
    return SequenceProcessor(SequenceConfig(target_frames=48))


def test_longer_sequence_sampled(processor):
    """Sequência maior que target_frames deve ser reduzida."""
    seq = np.random.randn(100, 346).astype(np.float32)
    out = processor.process(seq)
    assert out.shape == (48, 346)


def test_shorter_sequence_padded(processor):
    """Sequência menor que target_frames deve ser preenchida."""
    seq = np.random.randn(10, 346).astype(np.float32)
    out = processor.process(seq)
    assert out.shape == (48, 346)


def test_exact_sequence_unchanged(processor):
    """Sequência exata não deve alterar."""
    seq = np.random.randn(48, 346).astype(np.float32)
    out = processor.process(seq)
    assert out.shape == (48, 346)


def test_empty_sequence(processor):
    """Sequência vazia deve retornar zeros."""
    seq = np.zeros((0, 346), dtype=np.float32)
    out = processor.process(seq)
    assert out.shape == (48, 346)
    assert np.allclose(out, 0)


def test_mask_generation(processor):
    """Máscara deve indicar frames válidos."""
    seq = np.random.randn(20, 346).astype(np.float32)
    out = processor.process(seq)
    mask = processor.get_mask(original_len=20)
    assert len(mask) == 48
    assert sum(mask) > 0
