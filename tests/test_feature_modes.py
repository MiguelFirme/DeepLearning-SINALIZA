import numpy as np
import pytest

from ml.features.feature_modes import select_feature_mode
from ml.features.normalization import LandmarkNormalizer


def test_hands_only_ignores_pose_and_face_before_and_after_normalization():
    rng = np.random.default_rng(42)
    landmarks = rng.random((4, 346), dtype=np.float32)
    mask = np.ones((4, 4), dtype=bool)
    second = landmarks.copy()
    second[:, 126:] = rng.random((4, 220), dtype=np.float32)

    hands, hands_mask = select_feature_mode(landmarks, mask, "hands_only")
    other_hands, other_mask = select_feature_mode(second, mask, "hands_only")
    np.testing.assert_array_equal(hands[:, :126], landmarks[:, :126])
    np.testing.assert_array_equal(hands[:, 126:], 0)
    np.testing.assert_array_equal(hands_mask, np.array([[True, True, False, False]] * 4))
    np.testing.assert_array_equal(hands, other_hands)
    np.testing.assert_array_equal(hands_mask, other_mask)

    normalizer = LandmarkNormalizer()
    features = normalizer.normalize_sequence(hands, hands_mask)
    other_features = normalizer.normalize_sequence(other_hands, other_mask)
    np.testing.assert_array_equal(features, other_features)
    np.testing.assert_array_equal(features[:, 126:346], 0)
    np.testing.assert_array_equal(features[:, 346 + 126:692], 0)
    np.testing.assert_array_equal(features[:, 693:], 0)
    assert features.shape == (4, 697)


def test_hands_only_responds_to_hands_and_pose_face_still_zeros_hands():
    raw = np.ones((2, 346), dtype=np.float32)
    mask = np.ones((2, 4), dtype=bool)
    hand_features, _ = select_feature_mode(raw, mask, "hands_only")
    changed = raw.copy()
    changed[:, 0] = 2.0
    changed_hand_features, _ = select_feature_mode(changed, mask, "hands_only")
    assert not np.array_equal(hand_features, changed_hand_features)

    pose_face, pose_face_mask = select_feature_mode(raw, mask, "pose_face")
    np.testing.assert_array_equal(pose_face[:, :126], 0)
    np.testing.assert_array_equal(pose_face[:, 126:], 1)
    np.testing.assert_array_equal(pose_face_mask, np.array([[False, False, True, True]] * 2))
    np.testing.assert_array_equal(raw, 1)  # A selecao nao altera o cache.


def test_hands_only_requires_four_part_mask():
    with pytest.raises(ValueError, match="hands_only requer"):
        select_feature_mode(np.zeros((2, 346)), None, "hands_only")
