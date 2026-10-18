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

The window opens on a ready screen; click it and press SPACE.

## Why this project exists

We wanted a hand-controlled Dino-style game to get students interested in AI, using a pretrained hand model. The model worked: an open hand reliably produced a "jump" signal. But the game was unplayable for some people and boring for others. Fixing that turned out to be a data problem: physics, human timing, and a feedback loop. Most of this repo is about that part. It is a teaching example that a good model is not a good product.

## How it fits together

| Layer | Files | What they do |
|---|---|---|
| Perception | `gesture_lib.py`, `mp_compat.py` | Turn camera frames into a single "jump" event. |
| Game design | `adaptive.py` | Physics, difficulty profile, and obstacle rules. |
| Player modelling and control | `AdaptiveController`, `analyze.py`, `summarize.py` | Adapt d from 0/1 outcomes and report on behaviour. |

## Learning path

1. Run the notebook to see how landmarks become features.
2. Play `game.py` and watch `d` adapt.
3. Analyze your own logs with `analyze.py`.

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

Presets only set the starting difficulty (d, threshold, stability). The controller then adapts to your actual success rate, so the choice of preset matters mainly for the first few obstacles.

## Game over and `--max-crashes`

Crashes are counted since the game started or the preset changed. After `--max-crashes N` (default 3) the game shows on-screen stats and waits for SPACE to retry. Set `--max-crashes 0` to disable the stop. Games are short by default, so the controller adapts little within one; raise `--max-crashes` or use `0` for longer sessions.

## Data + bench commands

Gather a real session:

```bash
python game.py --log-outcomes data/alice.csv --log-latency data/alice_latency.csv --player alice --max-crashes 0
```

Use `--max-crashes 0` for a long run so the controller has time to adapt. Analyze with:

```bash
python analyze.py data/alice.csv
python summarize.py data/alice.csv
```

Run headless smoke or simulation:

```bash
SDL_VIDEODRIVER=dummy python game.py --dummy-camera --smoke-frames 120 --log-outcomes run.csv
python analyze.py run.csv

python dev_bench.py feasibility
python dev_bench.py report
python dev_bench.py simulate --frames 2000 --csv sim.csv
```

See all options with `python game.py --help`.

## Experiment mode: measuring a player

Instead of adaptive difficulty, the game can play a designed list of obstacles and record exactly when you jump. See [EXPERIMENT.md](EXPERIMENT.md) for what it measures and how to read the report.

```bash
python game.py --experiment --log-trials data/alice_trials.csv --player alice
python analyze_trials.py data/alice_trials.csv --plot data/alice_report.png

python simulate_experiment.py --mu 0.03 --sigma 0.05 --out sim_trials.csv   # no camera needed
```

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
