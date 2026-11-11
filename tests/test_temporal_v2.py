"""Tempo, janelas e metadados v2 com movimento sintético."""
import json

import numpy as np

from ml.features.temporal_v2 import resample_by_time
from ml.features.pipeline_v2 import prepare_features_v2
from ml.data.dataset_v2 import LibrasDatasetV2, window_end
from tests.test_pipeline_v2 import amostra


def test_reamostragem_interpolacao_mascara_e_bordas():
    raw, mask, _ = amostra(2)
    raw[1] += 1
    mask[1, 0] = False
    out, valid, grid = resample_by_time(raw, mask, [0, 1], 1.5, 2, 5)
    np.testing.assert_allclose(grid, [-.5, 0, .5, 1, 1.5])
    np.testing.assert_allclose(out[2, 63:126], (raw[0, 63:126] + raw[1, 63:126]) / 2)
    assert valid[2, 0] and not valid[-1, 0]
    assert not out[-1, :63].any()


def test_velocidade_nao_depende_do_fps_de_chegada():
    features = []
    for fps in (9, 30):
        raw, mask, _ = amostra(6 * fps + 1)
        times = np.arange(len(raw)) / fps
        raw[:, :126:3] += times[:, None] * .02
        values, _, _ = prepare_features_v2(raw, mask, times, 640, 480)
        features.append(values)
    np.testing.assert_allclose(features[0], features[1], atol=2e-5)


def test_dataset_v2_deterministico_e_avisa_metadata(tmp_path):
    raw, mask, _ = amostra(100, 30)
    np.savez(tmp_path / "sinal.npz", landmarks=raw, mask=mask)
    (tmp_path / "manifest.json").write_text(json.dumps([{"path": "sinal.npz", "label": "A", "signer": "P1"}]))
    ds = LibrasDatasetV2(tmp_path)
    assert ds.metadata_issues == {"fps": 1, "dimensions": 1}
    np.testing.assert_array_equal(ds[0]["sequence"], ds[0]["sequence"])
    assert ds[0]["sequence"].shape == (48, 725)


def test_janela_contem_80_porcento_do_trecho():
    rng = np.random.default_rng(5)
    times = np.arange(150) / 30
    for _ in range(100):
        end = window_end(times, 30, 90, 3.2, True, rng)
        overlap = min(end, times[89]) - max(end - 3.2, times[30])
        assert overlap >= .8 * (times[89] - times[30]) - 1e-8
