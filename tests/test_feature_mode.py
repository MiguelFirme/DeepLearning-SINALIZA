"""Ablação de mãos sem fabricar coordenadas nem alterar o cache."""
import json

import numpy as np
import pytest

from ml.data.dataset import LibrasDataset


def test_pose_face_masks_both_hands_before_normalization(tmp_path):
    landmarks = np.full((8, 346), 0.5, dtype=np.float32)
    landmarks[:, :126] = 0.9
    landmarks[:, 126:226:4] = 0.8
    landmarks[:, 130:226:4] = 0.2
    masks = np.ones((8, 4), dtype=bool)
    np.savez(tmp_path / "sample.npz", landmarks=landmarks, mask=masks)
    (tmp_path / "manifest.json").write_text(json.dumps([
        {"path": "sample.npz", "label": "Banheiro", "signer": "A"}
    ]), encoding="utf-8")
    full = LibrasDataset(tmp_path)
    pose_face = LibrasDataset(tmp_path, feature_mode="pose_face")
    full_seq = full[0]["sequence"].numpy()
    reduced = pose_face[0]["sequence"].numpy()
    assert full_seq.shape == reduced.shape == (48, 697)
    assert np.any(full_seq[:, :126] != 0)
    assert np.all(reduced[:, :126] == 0)
    assert np.all(reduced[:, 346:472] == 0)  # velocidade das mãos
    assert np.all(reduced[:, -5:] == 0)  # distâncias que dependem das mãos
    with np.load(tmp_path / "sample.npz") as cached:
        assert np.array_equal(cached["landmarks"], landmarks)
        assert np.array_equal(cached["mask"], masks)


def test_unknown_feature_mode_fails(tmp_path):
    with pytest.raises(ValueError, match="feature_mode"):
        LibrasDataset(tmp_path, feature_mode="invented_hands")
