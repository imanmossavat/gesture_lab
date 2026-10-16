"""Compatibility shim for MediaPipe legacy solutions.

Some wheels (e.g. MediaPipe 1.x / Python 3.13 on Apple Silicon) ship only the
Tasks API and no longer include ``mediapipe.python.solutions``.  This module
tries the legacy import first and falls back to no-op stubs so that the game
and the dev bench can still be imported and smoke-tested.

For real hand tracking, install a MediaPipe wheel that includes the legacy
solutions API (``mediapipe>=0.10.0,<1.0`` with a matching CPython/platform tag).
"""
try:
    import mediapipe as _mp

    mp_hands = _mp.solutions.hands
    mp_draw = _mp.solutions.drawing_utils
except (AttributeError, ImportError, ModuleNotFoundError):

    class _DummyResult:
        multi_hand_landmarks = None

    class _DummyHands:
        def __init__(self, **kwargs):
            pass

        def process(self, image):
            return _DummyResult()

        def close(self):
            pass

    class _DummyMpHands:
        Hands = _DummyHands
        HAND_CONNECTIONS = ()

    class _DummyDraw:
        @staticmethod
        def draw_landmarks(*args, **kwargs):
            pass

    mp_hands = _DummyMpHands()
    mp_draw = _DummyDraw()
