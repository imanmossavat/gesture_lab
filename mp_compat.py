"""Compatibility shim for MediaPipe hand tracking.

Some wheels (e.g. MediaPipe 0.10.x / Python 3.13 on Apple Silicon) ship only the
Tasks API and no longer include ``mediapipe.python.solutions``.  This module
tries the legacy import first and, if that fails, falls back to the Tasks API
using the ``hand_landmarker.task`` model downloaded on first use.

For offline use, pre-download the model to
``~/.cache/fontys_gesture_lab/hand_landmarker.task``.
"""

import os
import urllib.request


def _ensure_task_model() -> str:
    """Return the path to the hand_landmarker.task model, downloading if needed."""
    cache_dir = os.path.expanduser("~/.cache/fontys_gesture_lab")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "hand_landmarker.task")
    if os.path.exists(model_path):
        return model_path

    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    print(f"[mp_compat] downloading hand_landmarker.task to {model_path} ...")
    try:
        urllib.request.urlretrieve(url, model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download MediaPipe hand landmark model: {exc}\n"
            "Please download hand_landmarker.task manually and place it at:\n"
            f"  {model_path}"
        ) from exc
    return model_path


# ---------------------------------------------------------------------------
# Try legacy solutions first.
# ---------------------------------------------------------------------------
try:
    import mediapipe as _mp

    mp_hands = _mp.solutions.hands
    mp_draw = _mp.solutions.drawing_utils
except (AttributeError, ImportError, ModuleNotFoundError):
    # -----------------------------------------------------------------------
    # Legacy API unavailable -- use the Tasks API.
    # -----------------------------------------------------------------------
    import mediapipe as _mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    )

    class DrawingSpec:
        """Legacy-compatible drawing spec used by the Tasks fallback."""

        def __init__(self, color=(0, 0, 255), thickness=2, circle_radius=2):
            self.color = color
            self.thickness = thickness
            self.circle_radius = circle_radius

    def _norm_spec(spec):
        if spec is None:
            return DrawingSpec()
        if isinstance(spec, DrawingSpec):
            return spec
        return DrawingSpec(
            color=tuple(getattr(spec, "color", (0, 0, 255))),
            thickness=int(getattr(spec, "thickness", 2)),
            circle_radius=int(getattr(spec, "circle_radius", 2)),
        )

    def draw_landmarks(image, landmark_list, connections, landmark_drawing_spec=None,
                       connection_drawing_spec=None):
        """Render hand landmarks on a BGR OpenCV image."""
        import cv2

        if landmark_list is None or not hasattr(landmark_list, "landmark"):
            return
        landmarks = landmark_list.landmark
        if not landmarks:
            return

        h, w = image.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        lspec = _norm_spec(landmark_drawing_spec)
        cspec = _norm_spec(connection_drawing_spec)

        for start_idx, end_idx in connections or ():
            if start_idx < len(pts) and end_idx < len(pts):
                cv2.line(image, pts[start_idx], pts[end_idx], cspec.color, cspec.thickness)
        for pt in pts:
            cv2.circle(image, pt, lspec.circle_radius, lspec.color, -1)

    class _TaskMpDraw:
        DrawingSpec = DrawingSpec
        draw_landmarks = staticmethod(draw_landmarks)

    class _DummyResult:
        multi_hand_landmarks = None

    class _TaskHands:
        """Drop-in replacement for mediapipe.solutions.hands.Hands."""

        def __init__(
            self,
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,  # ignored by Tasks API
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ):
            model_path = _ensure_task_model()
            running_mode = RunningMode.IMAGE if static_image_mode else RunningMode.VIDEO
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=running_mode,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
            self._running_mode = running_mode
            self._frame_idx = 0

        def process(self, image):
            """Process an RGB numpy array and return a legacy-style result."""
            if image is None:
                return _DummyResult()

            mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=image)
            if self._running_mode == RunningMode.IMAGE:
                mp_result = self._landmarker.detect(mp_image)
            else:
                # 30 fps timestamp increment keeps each frame distinct.
                timestamp_ms = self._frame_idx * 33
                self._frame_idx += 1
                mp_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            if not mp_result or not mp_result.hand_landmarks:
                return _DummyResult()

            class _HandWrapper:
                def __init__(self, landmarks):
                    self.landmark = landmarks

            result = _DummyResult()
            result.multi_hand_landmarks = [_HandWrapper(hl) for hl in mp_result.hand_landmarks]
            return result

        def close(self):
            if hasattr(self._landmarker, "close"):
                self._landmarker.close()

    class _TaskMpHands:
        Hands = _TaskHands
        HAND_CONNECTIONS = HAND_CONNECTIONS

    mp_hands = _TaskMpHands()
    mp_draw = _TaskMpDraw()
