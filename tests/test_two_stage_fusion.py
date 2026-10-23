"""Checagens da restrição de candidatos e da ordem na fusão."""

import numpy as np
import pytest

from scripts.evaluate_two_stage_fusion import fuse_probabilities


def test_first_model_excludes_other_classes_even_when_second_is_confident():
    pose = np.array([[0.5, 0.4, 0.1]])
    full = np.array([[0.01, 0.01, 0.98]])
    scores, candidates = fuse_probabilities(pose, full, first="pose_face", top_k=2,
                                            pose_weight=0.65)
    assert candidates.tolist() == [[0, 1]]
    assert np.isneginf(scores[0, 2])
    assert scores.argmax(axis=1).tolist() == [0]


def test_reversing_first_model_can_change_result():
    pose = np.array([[0.5, 0.4, 0.1]])
    full = np.array([[0.01, 0.01, 0.98]])
    direct, _ = fuse_probabilities(pose, full, first="pose_face", top_k=1,
                                   pose_weight=0.65)
    reverse, _ = fuse_probabilities(pose, full, first="full", top_k=1,
                                    pose_weight=0.65)
    assert direct.argmax(axis=1).tolist() == [0]
    assert reverse.argmax(axis=1).tolist() == [2]


def test_combined_score_uses_fixed_model_weights():
    pose = np.array([[0.6, 0.4]])
    full = np.array([[0.2, 0.8]])
    scores, _ = fuse_probabilities(pose, full, first="pose_face", top_k=2,
                                   pose_weight=0.65)
    np.testing.assert_allclose(scores, [[0.46, 0.54]])


@pytest.mark.parametrize("top_k,weight", [(0, 0.65), (4, 0.65), (2, -0.1), (2, 1.1)])
def test_invalid_parameters_fail(top_k, weight):
    probs = np.array([[0.5, 0.3, 0.2]])
    with pytest.raises(ValueError):
        fuse_probabilities(probs, probs, first="pose_face", top_k=top_k,
                           pose_weight=weight)
