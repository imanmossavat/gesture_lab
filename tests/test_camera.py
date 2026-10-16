"""Hardware-dependent camera tests.

These are skipped automatically when OpenCV or a webcam is not available.
"""

import numpy as np
import pytest

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False


def _camera_available():
    if not CV2_AVAILABLE:
        return False
    cap = cv2.VideoCapture(0)
    available = cap.isOpened()
    cap.release()
    return available


CAMERA_AVAILABLE = _camera_available()

pytestmark = pytest.mark.skipif(
    not CAMERA_AVAILABLE,
    reason="No camera at index 0 or OpenCV not installed",
)


def test_default_camera_opens():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened()
    cap.release()


def test_camera_can_read_frame():
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    assert ok
    assert frame is not None
    assert len(frame.shape) == 3


def test_frame_is_not_all_black():
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    assert ok
    assert np.mean(frame) > 1.0
