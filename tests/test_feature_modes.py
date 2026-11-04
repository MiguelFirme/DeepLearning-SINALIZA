import numpy as np
import pytest

from ml.features.feature_modes import (
    FACE16_DROP_SLOTS, FACE16_KEEP_SLOTS, FACE16_MEDIAPIPE_INDICES,
    mask_normalized_face16, select_feature_mode,
)
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


@pytest.mark.parametrize("mode", ["pose_face16", "full_face16"])
def test_face16_keeps_only_selected_regions_even_after_normalization(mode):
    rng = np.random.default_rng(4)
    raw = rng.random((4, 346), dtype=np.float32)
    original = raw.copy()
    mask = np.ones((4, 4), dtype=bool)
    assert len(FACE16_MEDIAPIPE_INDICES) == len(FACE16_KEEP_SLOTS) == 16
    assert len(set(FACE16_KEEP_SLOTS)) == 16
    selected, selected_mask = select_feature_mode(raw, mask, mode)
    face = selected[:, 226:346].reshape(-1, 40, 3)
    np.testing.assert_array_equal(face[:, FACE16_DROP_SLOTS], 0)
    np.testing.assert_array_equal(face[:, FACE16_KEEP_SLOTS],
                                  raw[:, 226:346].reshape(-1, 40, 3)[:, FACE16_KEEP_SLOTS])
    np.testing.assert_array_equal(raw, original)
    if mode == "pose_face16":
        np.testing.assert_array_equal(selected[:, :126], 0)
        assert not selected_mask[:, :2].any()
    else:
        np.testing.assert_array_equal(selected[:, :126], raw[:, :126])

    normalized = LandmarkNormalizer().normalize_sequence(selected, selected_mask)
    # O normalizador centraliza zeros pelo ombro; a máscara final precisa removê-los.
    assert np.any(normalized[:, 226:346].reshape(-1, 40, 3)[:, FACE16_DROP_SLOTS] != 0)
    final = mask_normalized_face16(normalized, mode)
    for start in (226, 346 + 226):
        np.testing.assert_array_equal(final[:, start:start + 120].reshape(-1, 40, 3)[:, FACE16_DROP_SLOTS], 0)
    assert final.shape == (4, 697)


def test_face16_reapply_after_augmentation_clears_excluded_points():
    raw = np.ones((2, 346), dtype=np.float32)
    mask = np.ones((2, 4), dtype=bool)
    selected, selected_mask = select_feature_mode(raw, mask, "full_face16")
    selected[:, 226:346] += 0.05  # simula augmentation que altera até os zeros
    cleaned, _ = select_feature_mode(selected, selected_mask, "full_face16")
    np.testing.assert_array_equal(cleaned[:, 226:346].reshape(-1, 40, 3)[:, FACE16_DROP_SLOTS], 0)
