#  Gesture Lab

A small, educational hand-gesture project for students.

It contains three parts that share the same gesture code:

1. **`game.py`** — a runner controlled by an open hand, now with adaptive difficulty.
2. **`dev_bench.py`** — a headless bench for tuning the difficulty system without a camera.
3. **`gesture_landmarks_lab.ipynb`** — a guided Jupyter notebook that shows how MediaPipe landmarks are detected, normalised, and turned into a gesture decision.

No model training is required. The classifier is a small set of hand-written rules.

## What students learn

- A pretrained model (MediaPipe Hands) gives us 21 landmark points.
- Those points are normalised so the classifier is less sensitive to hand position and size.
- A rule-based classifier decides whether fingers are extended.
- A score is not the same as guaranteed correctness.
- Difficulty can be driven by physics and a target success rate.
- Real apps must handle camera differences, lighting, rotation, latency, and different people.

## Requirements

- Python 3.10, 3.11, 3.12, or 3.13
- A webcam (only for playing with real gestures)
- A computer with a screen (only for the normal game window)
- **Not Google Colab**: the live camera cells and Pygame game do not work there.

## Setup

Create a virtual environment, install the dependencies, and run the tests.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python game.py
```

### Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python game.py
```

## Platform notes

- This project was developed on macOS. The same packages work on Windows and Linux.
- If `cv2.VideoCapture(0)` cannot open your webcam, try index `1`:
  ```bash
  python game.py --camera 1
  ```
- On some Macs you may see a harmless SDL warning when OpenCV and Pygame are imported together. The test suite includes a smoke test to make sure this does not crash the program.
- The normal game needs a display, but a headless smoke test is supported via `SDL_VIDEODRIVER=dummy` (see below).
- Some MediaPipe wheels (e.g. Python 3.13 on Apple Silicon) ship only the Tasks API. `mp_compat.py` lets the game run in smoke mode; for real hand tracking on those wheels, install a legacy-solutions MediaPipe build.

## Game controls

| Input | Action |
|---|---|
| **Open hand** | Jump (once per stable gesture) |
| **1** | Easy preset (d = 0.25) |
| **2** | Normal preset (d = 0.50) |
| **3** | Hard preset (d = 0.75) |
| **+ / -** | Raise / lower confidence threshold |
| **R** | Restart |
| **D** | Toggle debug overlay |
| **Q** or **Esc** | Quit |

The jump fires only once per stable open-hand gesture. Close your hand to re-arm it.

## Adaptive difficulty

The game now uses a physics-informed adaptive controller instead of fixed speed tiers.

- `d` is a normalised difficulty in `[0, 1]`.
- Speed, obstacle height/width, and spawn interval are derived from `d`.
- The controller targets a 75% success rate:
  `d <- clip(d + 0.08 * (outcome - 0.75))`
- A minimum physical gap rule guarantees every spawned obstacle is theoretically
  jumpable, even at maximum difficulty.

See `DEV.md` for the full design, tuning notes, and formulas.

## Difficulty presets

| Preset | d | Stability frames | Threshold |
|---|---|---|---|
| Easy | 0.25 | 1 | 0.50 |
| Normal | 0.50 | 2 | 0.60 |
| Hard | 0.75 | 3 | 0.70 |

## Headless smoke test

No camera or display is required:

```bash
SDL_VIDEODRIVER=dummy python game.py --dummy-camera --smoke-frames 120
```

You can also log latency without opening a window:

```bash
SDL_VIDEODRIVER=dummy python game.py --dummy-camera --smoke-frames 500 --log-latency latency.csv
```

## Dev bench

`dev_bench.py` exercises the adaptive logic without Pygame, OpenCV, or MediaPipe:

```bash
python dev_bench.py feasibility
python dev_bench.py sweep
python dev_bench.py simulate --frames 2000 --noise 80 --latency 90
python dev_bench.py compare --output compare.png
python dev_bench.py latency-report --max-latency 200 --output latency.png
python dev_bench.py play   # explains that play mode lives in game.py
```

## Notebook

Open `gesture_landmarks_lab.ipynb` in VS Code or Jupyter:

```bash
jupyter notebook gesture_landmarks_lab.ipynb
```

The notebook covers:

1. Opening the webcam and capturing a frame.
2. Detecting and drawing the 21 MediaPipe landmarks.
3. Normalising landmarks: translating the wrist to the origin and scaling by hand size.
4. Building a 63-number feature vector.
5. How the rule-based classifier decides between open, fist, and custom.
6. The effect of the confidence threshold.
7. Failure cases and reflection questions.

## Running tests

```bash
pytest -q
```

- `tests/test_gesture_lib.py` checks the landmark helpers and rule classifier with synthetic data. It runs on any machine.
- `tests/test_adaptive.py` checks the difficulty profile, controller, obstacle resolution, min-gap rule, and physics helpers. It runs on any machine.
- `tests/test_camera.py` checks the webcam. It is automatically skipped if no camera is available.

## Project files

- `game.py` — gesture-controlled runner with adaptive difficulty.
- `adaptive.py` — pure-Python adaptive difficulty logic.
- `mp_compat.py` — MediaPipe legacy-solutions compatibility shim.
- `dev_bench.py` — headless tuning bench.
- `gesture_lib.py` — landmark conversion, normalisation, and rule-based classifier.
- `gesture_landmarks_lab.ipynb` — educational notebook.
- `tests/` — pytest tests.
- `requirements.txt` — Python dependencies.
- `DEV.md` — developer / tuning guide.
- `README.md` — this file.

## Troubleshooting

- **Camera unavailable:** close Teams/Zoom/browser tabs using the webcam and check OS camera permissions. Try `--camera 1`.
- **No hand detected:** improve lighting, show the full hand, and move closer to the camera.
- **Slow laptop:** lower the camera resolution or choose the Easy preset.
- **Tests fail on import:** make sure you are inside the virtual environment and `pytest` is installed.
- **Headless smoke crashes with display errors:** set `SDL_VIDEODRIVER=dummy` before the command.
