"""Revisão 2026-11-13: segmentação compartilhada e features sem profundidade global."""
import numpy as np

from ml.features.pipeline_v2 import PipelineV2Config, build_features_v2
from ml.features.segmentation_v2 import centered_window_end, main_segment, motion_segments


def gesto(fps, repouso=1.5, sinal=1.2, depois=2.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int((repouso + sinal + depois) * fps)) / fps
    pose = np.zeros((len(t), 25, 4), np.float32)
    pose[:, 5, :2], pose[:, 6, :2], pose[:, :, 3] = [.4, .4], [.6, .4], 1
    fase = np.clip((t - repouso) / sinal, 0, 1) * ((t >= repouso) & (t < repouso + sinal))
    pose[:, 9, 0], pose[:, 9, 1] = .42, .8 - .25 * np.sin(np.pi * fase)
    pose[:, 10, :2] = [.58, .8]
    pose[:, :, :2] += rng.normal(0, .003, pose[:, :, :2].shape)
    raw = np.zeros((len(t), 346), np.float32)
    raw[:, 126:226] = pose.reshape(len(t), 100)
    mask = np.ones((len(t), 4), bool)
    mask[:, :2] = False
    return raw, mask, t


def test_segmento_igual_em_fps_diferentes():
    centros = []
    for fps in (6, 9, 15, 30):
        segmento = main_segment(*gesto(fps), 640, 480)
        assert segmento is not None
        assert 1.3 < segmento[0] < 1.95 and 2.5 < segmento[1] < 3.0
        centros.append(centered_window_end(segmento, 3.2))
    assert max(centros) - min(centros) < .25


def test_repouso_nao_gera_segmento():
    raw, mask, t = gesto(15, sinal=0.0001)
    assert motion_segments(raw, mask, t, 640, 480, require_closed=False) == []


def test_profundidade_da_pose_nao_altera_features_revisadas():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 1, (48, 346)).astype(np.float32)
    mask = np.ones((48, 4), bool)
    t = np.arange(48) / 15
    outro = raw.copy()
    outro[:, 126:226].reshape(48, 25, 4)[:, :, 2] -= .7
    cfg = PipelineV2Config(global_depth=False, scale_2d=True)
    assert np.allclose(build_features_v2(raw, mask, t, 640, 480, cfg),
                       build_features_v2(outro, mask, t, 640, 480, cfg))
    # O padrão continua igual ao checkpoint já treinado (sensível ao z).
    assert not np.allclose(build_features_v2(raw, mask, t, 640, 480),
                           build_features_v2(outro, mask, t, 640, 480))


def test_pontos_do_braco_fora_do_quadro_sao_zerados():
    rng = np.random.default_rng(2)
    raw = rng.uniform(.2, .8, (48, 346)).astype(np.float32)
    raw[:, 126:226].reshape(48, 25, 4)[:, 7:17, 3] = .1   # cotovelos/punhos invisíveis
    mask = np.ones((48, 4), bool)
    t = np.arange(48) / 15
    features = build_features_v2(raw, mask, t, 640, 480, PipelineV2Config(point_visibility=True))
    assert np.all(features[:, 126:177].reshape(48, 17, 3)[:, 7:] == 0)
