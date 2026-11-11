"""Invariâncias físicas e ausência de partes no construtor v2."""
import numpy as np

from ml.features.pipeline_v2 import PipelineV2Config, build_features_v2


def amostra(n=48, fps=15):
    raw = np.zeros((n, 346), np.float32)
    pose = raw[:, 126:226].reshape(n, 25, 4)
    pose[:, 5, :3] = [.3, .4, 0]
    pose[:, 6, :3] = [.7, .4, 0]
    for side in (0, 1):
        hand = raw[:, side * 63:(side + 1) * 63].reshape(n, 21, 3)
        hand[:] = np.arange(21)[None, :, None] * .003 + .2
    return raw, np.ones((n, 4), bool), np.arange(n) / fps


def test_aspecto_e_remocao_do_atalho():
    raw, mask, times = amostra()
    other = raw.copy()
    for start, end, step in ((0, 126, 3), (126, 226, 4), (226, 346, 3)):
        other[:, start:end:step] *= (640 / 480) / (1920 / 1080)
    a = build_features_v2(raw, mask, times, 640, 480)
    b = build_features_v2(other, mask, times, 1920, 1080)
    np.testing.assert_allclose(a, b, atol=1e-5)
    other = raw.copy()
    other[:, 126 + 17 * 4:226] = 500
    other[:, 129:226:4] = .01
    np.testing.assert_allclose(a, build_features_v2(other, mask, times, 640, 480))
    assert a.shape == (48, 725)


def test_forma_local_invariante_e_lacunas():
    raw, mask, times = amostra()
    a = build_features_v2(raw, mask, times, 640, 480)
    other = raw.copy()
    other[:, :126] = other[:, :126] * 1.3 + .1
    b = build_features_v2(other, mask, times, 640, 480)
    np.testing.assert_allclose(a[:, 297:423], b[:, 297:423], atol=1e-5)
    mask[5:8, 0] = False
    mask[15:23, 0] = False
    raw[~mask[:, 0], :63] = 0
    result = build_features_v2(raw, mask, times, 640, 480, PipelineV2Config(max_gap=5))
    assert np.any(result[6, :63])
    assert not np.any(result[15:23, :63])
    assert not np.any(result[15:23, 297:360])
    assert not np.any(result[15:24, 428:491])
