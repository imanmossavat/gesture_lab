# Gesture Lab

A small, educational hand-gesture project for students. Play a runner with an open-hand jump, tune the adaptive difficulty, and analyze the logs.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
python game.py
```

## Why this project exists

- Show how a pretrained model (MediaPipe Hands) gives 21 landmark points.
- Turn those points into a gesture decision with a small rule-based classifier.
- Drive difficulty with physics and a target success rate instead of hand-coded levels.
- Make students think about latency, lighting, camera differences, and real-world robustness.

## How it fits together

| File | What it does |
|---|---|
| `game.py` | Play the gesture-controlled runner. |
| `dev_bench.py` | Tune the adaptive logic headlessly. |
| `analyze.py` / `summarize.py` | Read CSV logs and print statistics or plots. |
| `gesture_landmarks_lab.ipynb` | Guided notebook on landmarks and classification. |
| `adaptive.py` | Pure difficulty, physics, controller, and CSV headers. |
| `gesture_lib.py` | Landmark helpers and rule classifier. |
| `mp_compat.py` | MediaPipe legacy `solutions` → Tasks API fallback. |

## Learning path

1. Run the notebook to see how landmarks become features.
2. Play `game.py` and watch `d` adapt.
3. Run `python dev_bench.py report` to inspect the design numbers.
4. Change `Config` values and rerun `dev_bench.py feasibility`.
5. Analyze your own logs with `analyze.py`.

## Controls

| Input | Action |
|---|---|
| **SPACE** | Start / pause / resume / retry |
| **Open hand** | Jump (once per stable gesture) |
| **1 / 2 / 3** | Easy / normal / hard preset |
| **+ / -** | Raise / lower confidence threshold |
| **D** | Toggle debug overlay |
| **Q** or **Esc** | Quit |

Close your hand to re-arm the jump. Preset changes clear obstacles and start a new session.

## Difficulty presets

| Preset | d | Stability | Threshold |
|---|---|---|---|
| easy | 0.80 | 1 | 0.50 |
| normal | 0.94 | 2 | 0.60 |
| hard | 1.00 | 3 | 0.70 |

The controller still adapts within a preset; presets set the starting difficulty.

## Game over and `--max-crashes`

Crashes are counted per session. After `--max-crashes N` (default 3) the game shows on-screen stats and waits for SPACE to retry. Set `--max-crashes 0` to disable the stop.

## Data + bench commands

```bash
SDL_VIDEODRIVER=dummy python game.py --dummy-camera --smoke-frames 120 --log-outcomes run.csv
python analyze.py run.csv
python summarize.py run.csv

python dev_bench.py feasibility
python dev_bench.py report
python dev_bench.py simulate --frames 2000 --csv sim.csv
```

Bench CSVs include a metadata header with the full `Config` so reports stay reproducible.

## Tests

```bash
pytest -q
```

- `tests/test_gesture_lib.py`: landmark helpers and rule classifier.
- `tests/test_adaptive.py`: difficulty, controller, physics, obstacle resolution, phase flow, and game-loop smoke tests.
- `tests/test_camera.py`: webcam checks, skipped if no camera.

## Platform notes

- Python 3.10–3.13 are supported.
- macOS, Windows, and Linux should work; the project was developed on macOS.
- If `cv2.VideoCapture(0)` cannot open the webcam, try `--camera 1`.
- Some MediaPipe wheels (Python 3.13 / Apple Silicon) ship only the Tasks API; `mp_compat.py` falls back automatically.
- The normal game needs a display; headless smoke runs work with `SDL_VIDEODRIVER=dummy`.

## Troubleshooting

- **Window not visible:** it may open behind other windows. Click the game icon in the dock/taskbar.
- **Camera unavailable:** close other apps using the webcam and check OS permissions; try `--camera 1`.
- **No hand detected:** improve lighting, show the full hand, move closer.
- **Slow laptop:** lower camera resolution or choose the Easy preset.
- **Tests fail on import:** activate the virtual environment and install `requirements.txt`.
- **Headless smoke crashes:** set `SDL_VIDEODRIVER=dummy`.
