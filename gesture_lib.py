"""Shared hand-landmark utilities and the rule-based gesture classifier.

The classifier works with MediaPipe's 21 hand landmarks in the order documented at
https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer
"""

import numpy as np

GESTURES = ("open", "fist", "custom")
JUMP_GESTURES = ("open", "thumb")     # which pose triggers a jump (game.py --gesture)

# Friendly names for the 21 landmarks, mainly for teaching and diagrams.
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


def landmarks_to_array(hand_landmarks):
    """Accept a (21, 3) ndarray/list or a MediaPipe NormalizedLandmarkList."""
    if isinstance(hand_landmarks, np.ndarray):
        arr = np.asarray(hand_landmarks, dtype=np.float32)
    elif isinstance(hand_landmarks, (list, tuple)):
        arr = np.asarray(
            [[p.x, p.y, p.z] if hasattr(p, "x") else p for p in hand_landmarks],
            dtype=np.float32,
        )
    else:
        points = getattr(hand_landmarks, "landmark", hand_landmarks)
        arr = np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected landmarks shape (21, 3), got {arr.shape}")
    return arr


def normalize_landmarks(landmarks):
    """Translate wrist to origin and scale by hand size.

    This removes much of the sensitivity to hand position and distance from the
    camera. It does not remove all variation such as rotation, camera angle, or
    differences between people.
    """
    points = landmarks_to_array(landmarks).copy()
    points -= points[0]  # landmark 0 is the wrist
    scale = np.linalg.norm(points[:, :2], axis=1).max()
    if scale > 1e-6:
        points /= scale
    return points.astype(np.float32)


def feature_vector(landmarks):
    """Flatten normalized 21 x 3 coordinates into a 63-number vector."""
    return normalize_landmarks(landmarks).reshape(-1)


def _finger_extended(points, tip, pip):
    """Return True if the fingertip is above the PIP joint.

    In image coordinates y grows downward, so a smaller y means higher up.
    This works best when the palm is roughly facing the camera.
    """
    return points[tip, 1] < points[pip, 1] - 0.025


class RuleClassifier:
    """A transparent, hand-written gesture classifier, not a learned model."""

    def classify(self, landmarks):
        """Return (label, score)."""
        p = landmarks_to_array(landmarks)
        fingers = [
            _finger_extended(p, 8, 6),   # index
            _finger_extended(p, 12, 10),  # middle
            _finger_extended(p, 16, 14),  # ring
            _finger_extended(p, 20, 18),  # pinky
        ]
        extended = sum(fingers)
        # The thumb is intentionally ignored because its direction changes a lot
        # with hand orientation.
        if extended >= 3:
            return "open", min(0.99, 0.55 + 0.11 * extended)
        if extended == 0:
            return "fist", 0.75
        return "custom", 0.55 + 0.10 * (3 - extended)


# A raised thumb must clear the index knuckle by this fraction of the hand size
# (wrist to middle knuckle).  Lower it if your thumb is not picked up.
THUMB_RISE_MIN = 0.35


class ThumbClassifier(RuleClassifier):
    """Like RuleClassifier, plus "thumb": a fist with the thumb raised.

    Rest pose is a fist (thumb tucked).  Raising the thumb gives "thumb", so the
    jump gesture is thumb up, and lowering it re-arms.  Like the finger test this
    assumes the hand is roughly upright in the image (y grows downward).
    """

    def classify(self, landmarks):
        label, score = super().classify(landmarks)
        if label != "fist":
            return label, score
        p = landmarks_to_array(landmarks)
        hand = float(np.linalg.norm(p[9, :2] - p[0, :2]))      # wrist to middle knuckle
        if hand < 1e-6:
            return label, score
        rise = (p[5, 1] - p[4, 1]) / hand        # thumb tip above the index knuckle
        above_ip = (p[3, 1] - p[4, 1]) / hand    # and above its own joint: pointing up
        if rise > THUMB_RISE_MIN and above_ip > 0.05:
            return "thumb", float(min(0.99, 0.60 + (rise - THUMB_RISE_MIN)))
        return label, score


def make_classifier(gesture: str = "open"):
    """The classifier for a jump gesture: "open" (open hand) or "thumb" (thumb up)."""
    if gesture not in JUMP_GESTURES:
        raise ValueError(f"gesture must be one of {JUMP_GESTURES}, got {gesture!r}")
    return ThumbClassifier() if gesture == "thumb" else RuleClassifier()
