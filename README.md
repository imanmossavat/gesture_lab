#  Gesture Lab

A small, educational hand-gesture project for students.

It contains four parts that share the same gesture code:

1. **`game.py`** — a runner controlled by an open hand, now with adaptive difficulty.
2. **`dev_bench.py`** — a headless bench for tuning the difficulty system without a camera.
3. **`analyze.py`** — reads latency or outcome CSVs and prints per-session statistics and plots.
4. **`gesture_landmarks_lab.ipynb`** — a guided Jupyter notebook that shows how MediaPipe landmarks are detected, normalised, and turned into a gesture decision.

No model training is required. The classifier is a small set of hand-written rules.

## What students learn

- A pretrained model (MediaPipe Hands) gives us 21 landmark points.
- Those points can be normalised into a 63-number feature vector; the rule classifier uses raw landmarks.
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
- Some MediaPipe wheels (e.g. Python 3.13 on Apple Silicon) ship only the Tasks API. `mp_compat.py` automatically uses the Tasks API `HandLandmarker` in that case and downloads the required `hand_landmarker.task` model on first use.

## Game controls

| Input | Action |
|---|---|
| **Open hand** | Jump (once per stable gesture) |
| **1** | Easy preset (d = 0.80) |
| **2** | Normal preset (d = 0.94) |
| **3** | Hard preset (d = 1.00) |
| **+ / -** | Raise / lower confidence threshold |
| **R** | Restart |
| **D** | Toggle debug overlay |
| **Q** or **Esc** | Quit |

The jump fires only once per stable open-hand gesture. Close your hand to re-arm it.

Preset changes (**1/2/3**) and restarts (**R**) also start a new session: obstacles are cleared, the spawn timer is reset, and the `session` column in the outcome log increments.

## Adaptive difficulty

`d` is a normalised difficulty in `[0, 1]`.  Speed, obstacle geometry, and spawn
interval are derived from `d` by `adaptive.difficulty_profile`.  The
`AdaptiveController` targets `Config.TARGET_SUCCESS_RATE` and supports optional
momentum; see `adaptive.py` and `python dev_bench.py report` for the exact
numbers.  A minimum physical gap rule guarantees every spawned obstacle is
theoretically jumpable.


## Difficulty presets

| Preset | d | Stability frames | Threshold |
|---|---|---|---|
| Easy | 0.80 | 1 | 0.50 |
| Normal | 0.94 | 2 | 0.60 |
| Hard | 1.00 | 3 | 0.70 |

## Collecting data

The game can write every obstacle outcome to a CSV file for later analysis.

```bash
mkdir -p data
python game.py --log-outcomes data/alice_2026-10-16.csv --player alice
```

The log is written continuously; press **Q** or **Esc** to stop. Each row records:

```text
t_s, player, session, mode, d_spawn, d_after, speed, height, width, outcome
```

- `session` increments every time you press **1/2/3** or **R**, so each preset block or restart becomes its own session.
- `mode` is the preset used (1 = easy, 2 = normal, 3 = hard).
- `d_spawn` is the difficulty `d` when the obstacle was spawned.
- `outcome` is `1` for a clear and `0` for a crash.

Analyze the file with:

```bash
python analyze.py data/alice_2026-10-16.csv
python analyze.py data/alice_2026-10-16.csv --plot report.png
```

For multi-session files `--plot` writes one image per session (`report_alice_2026-10-16_s1.png`, `report_alice_2026-10-16_s2.png`, ...) plus a pooled text summary. Trim noisy starts/ends with:

```bash
python analyze.py data/alice_2026-10-16.csv --trim-start 3 --trim-end 3
python analyze.py data/alice_2026-10-16.csv --trim-start-s 10 --trim-end-s 10
```

See `DEV.md` for the full log format and tuning notes.

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
python dev_bench.py report
python dev_bench.py sweep
python dev_bench.py simulate --frames 2000 --noise 80 --latency 90
python dev_bench.py compare --output compare.png
python dev_bench.py latency-report --max-latency 200 --output latency.png
python dev_bench.py play   # explains that play mode lives in game.py
```

Bench CSVs written with `--csv` include a metadata header containing the full
`Config` used, so `analyze.py` and `summarize.py` can show which tuning
parameters produced the numbers.

## Notebook

Open `gesture_landmarks_lab.ipynb` in VS Code or Jupyter:

```bash
jupyter notebook gesture_landmarks_lab.ipynb
```

The notebook covers:

1. Opening the webcam and capturing a frame.
2. Detecting and drawing the 21 MediaPipe landmarks.
3. Normalising landmarks into an optional 63-number feature vector (the rule classifier uses raw landmarks).
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
- `mp_compat.py` — MediaPipe compatibility shim (legacy solutions or Tasks API).
- `dev_bench.py` — headless tuning bench.
- `analyze.py` — reads latency / outcome CSVs and prints reports and plots.
- `gesture_lib.py` — landmark conversion, normalisation, and rule-based classifier.
- `gesture_landmarks_lab.ipynb` — educational notebook.
- `tests/` — pytest tests.
- `requirements.txt` — Python dependencies.
- `AGENTS.md` — slow-changing reference for agents / contributors.
- `DEV.md` — developer / tuning guide.
- `README.md` — this file.

## Troubleshooting

- **Camera unavailable:** close Teams/Zoom/browser tabs using the webcam and check OS camera permissions. Try `--camera 1`.
- **No hand detected:** improve lighting, show the full hand, and move closer to the camera.
- **Slow laptop:** lower the camera resolution or choose the Easy preset.
- **Tests fail on import:** make sure you are inside the virtual environment and `pytest` is installed.
- **Headless smoke crashes with display errors:** set `SDL_VIDEODRIVER=dummy` before the command.
