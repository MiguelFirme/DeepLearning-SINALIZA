import numpy as np
import pytest

from sinaliza.sequencias import redimensionar_sequencia


def test_redimensionamento_preserva_extremos() -> None:
    sequence = np.asarray([[0.0, 2.0], [10.0, 12.0]], dtype=np.float32)

    resized = redimensionar_sequencia(sequence, 5)

    assert resized.shape == (5, 2)
    np.testing.assert_array_equal(resized[0], sequence[0])
    np.testing.assert_array_equal(resized[-1], sequence[-1])


def test_redimensionamento_repete_quadro_unico() -> None:
    resized = redimensionar_sequencia(np.asarray([[1.0, 2.0]]), 3)

    np.testing.assert_array_equal(resized, [[1.0, 2.0]] * 3)


def test_redimensionamento_rejeita_entrada_vazia() -> None:
    with pytest.raises(ValueError):
        redimensionar_sequencia(np.empty((0, 2)), 10)
