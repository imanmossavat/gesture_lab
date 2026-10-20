"""Tests for the rule-based gesture library.

These tests are pure logic and should pass on any machine, with or without a camera.
"""

import numpy as np
import pytest

from gesture_lab.gesture_lib import (
    GESTURES,
    RuleClassifier,
    feature_vector,
    landmarks_to_array,
    normalize_landmarks,
)


@pytest.fixture
def flat_hand():
    """A hand with all four fingers clearly extended (thumb ignored)."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[8] = [0.2, 0.1, 0.0]   # index tip
    pts[6] = [0.2, 0.4, 0.0]   # index pip
    pts[12] = [0.3, 0.1, 0.0]  # middle tip
    pts[10] = [0.3, 0.4, 0.0]  # middle pip
    pts[16] = [0.4, 0.1, 0.0]  # ring tip
    pts[14] = [0.4, 0.4, 0.0]  # ring pip
    pts[20] = [0.5, 0.1, 0.0]  # pinky tip
    pts[18] = [0.5, 0.4, 0.0]  # pinky pip
    return pts


@pytest.fixture
def fist():
    """A hand with all four fingers folded."""
    pts = np.zeros((21, 3), dtype=np.float32)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        pts[pip] = [0.2, 0.2, 0.0]
        pts[tip] = [0.2, 0.5, 0.0]
    return pts


def test_gestures_tuple_contains_expected_labels():
    assert set(GESTURES) == {"open", "fist", "custom"}


def test_landmarks_to_array_accepts_numpy_array():
    arr = np.random.rand(21, 3).astype(np.float32)
    out = landmarks_to_array(arr)
    assert out.shape == (21, 3)
    assert np.allclose(out, arr)


def test_landmarks_to_array_accepts_list_of_tuples():
    data = [(i * 0.1, i * 0.2, 0.0) for i in range(21)]
    out = landmarks_to_array(data)
    assert out.shape == (21, 3)
    assert out.dtype == np.float32


def test_landmarks_to_array_rejects_wrong_shape():
    with pytest.raises(ValueError):
        landmarks_to_array(np.zeros((20, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        landmarks_to_array(np.zeros((21, 2), dtype=np.float32))


def test_normalize_landmarks_puts_wrist_at_origin():
    pts = np.ones((21, 3), dtype=np.float32)
    pts[:, 0] = np.arange(21) * 0.1
    norm = normalize_landmarks(pts)
    assert np.allclose(norm[0], [0.0, 0.0, 0.0])


def test_normalize_landmarks_scales_hand_to_unit_size():
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[:, 0] = np.arange(21) * 2.0
    norm = normalize_landmarks(pts)
    max_dist = np.linalg.norm(norm[:, :2], axis=1).max()
    assert np.isclose(max_dist, 1.0, atol=1e-5)


def test_feature_vector_has_63_values():
    pts = np.random.rand(21, 3).astype(np.float32)
    vec = feature_vector(pts)
    assert vec.shape == (63,)
    assert vec.dtype == np.float32


def test_rule_classifier_detects_open_hand(flat_hand):
    label, score = RuleClassifier().classify(flat_hand)
    assert label == "open"
    assert 0.0 <= score <= 1.0


def test_rule_classifier_detects_fist(fist):
    label, score = RuleClassifier().classify(fist)
    assert label == "fist"
    assert 0.0 <= score <= 1.0


def test_rule_classifier_detects_custom():
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[8] = [0.2, 0.1, 0.0]
    pts[6] = [0.2, 0.4, 0.0]
    for tip, pip in [(12, 10), (16, 14), (20, 18)]:
        pts[pip] = [0.3, 0.2, 0.0]
        pts[tip] = [0.3, 0.5, 0.0]
    label, score = RuleClassifier().classify(pts)
    assert label == "custom"
    assert 0.0 <= score <= 1.0


def test_rule_classifier_score_is_between_zero_and_one(flat_hand, fist):
    cls = RuleClassifier()
    for pts in (flat_hand, fist):
        _, score = cls.classify(pts)
        assert 0.0 <= score <= 1.0


def test_importing_opencv_and_pygame_together_does_not_crash():
    """Smoke test for the SDL duplicate-symbol warning sometimes seen on macOS."""
    pytest.importorskip("cv2")
    pytest.importorskip("pygame")
    import cv2  # noqa: F401
    import pygame  # noqa: F401


# --- thumb-up gesture ---------------------------------------------------------

def _hand(fingers_up, thumb):
    """A plausible upright hand (image y grows downward).  thumb: up | tucked | side."""
    p = np.zeros((21, 3), dtype=np.float32)
    p[0, :2] = [0.50, 0.80]                               # wrist
    for (mcp, pip, tip), x in zip([(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)],
                                  [0.45, 0.50, 0.55, 0.60]):
        p[mcp, :2] = [x, 0.57 if mcp == 5 else 0.55]
        p[pip, :2] = [x, 0.50]
        p[tip, :2] = [x, 0.35] if fingers_up else [x, 0.56]
    p[1, :2], p[2, :2], p[3, :2] = [0.43, 0.75], [0.40, 0.68], [0.38, 0.64]
    p[4, :2] = {"up": [0.37, 0.45], "tucked": [0.44, 0.58], "side": [0.25, 0.62]}[thumb]
    return p


def test_thumb_classifier_detects_thumb_up_from_fist():
    from gesture_lab.gesture_lib import ThumbClassifier
    label, score = ThumbClassifier().classify(_hand(False, "up"))
    assert label == "thumb" and 0.6 <= score <= 0.99


def test_thumb_classifier_keeps_fist_and_other_poses():
    from gesture_lab.gesture_lib import ThumbClassifier
    c = ThumbClassifier()
    assert c.classify(_hand(False, "tucked"))[0] == "fist"
    assert c.classify(_hand(False, "side"))[0] == "fist"      # thumb out sideways is not thumb up
    assert c.classify(_hand(True, "up"))[0] == "open"         # an open hand never counts as thumb


def test_rule_classifier_never_returns_thumb():
    assert RuleClassifier().classify(_hand(False, "up"))[0] == "fist"


def test_make_classifier_selects_by_name_and_rejects_unknown():
    from gesture_lab.gesture_lib import ThumbClassifier, make_classifier
    assert type(make_classifier("open")) is RuleClassifier
    assert isinstance(make_classifier("thumb"), ThumbClassifier)
    with pytest.raises(ValueError):
        make_classifier("wave")
