# Gesture Lab

A hand-controlled runner that doubles as a small experiment: how precisely can you time a jump with a hand gesture? Play it, then see a report on your own timing.

## Play in five minutes

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python game.py
```

A window opens (it may hide behind other windows). Click it and press **SPACE**.

- **First run needs internet once.** On some installs the hand model (`hand_landmarker.task`) downloads on first use. For offline use, place it at `~/.cache/fontys_gesture_lab/hand_landmarker.task`.
- **Setup:** sit about an arm's length from the camera, hand fully in view, fingers pointing up, good light.
- **Jump:** open your hand. **Close it again** before the next jump (that re-arms it).

## Pick a mode

| I want to... | Command |
|---|---|
| Just play | `python game.py` |
| Jump with a thumb instead of an open hand | `python game.py --gesture thumb` (thumb up from a fist; lower it to re-arm) |
| Be measured: a fixed list of 31 obstacles, same for everyone with the same seed | `python game.py --experiment --experiment-seed 42 --player alice --log-trials data/alice_1.csv` |
| A progressive challenge: it finds your narrowest feasible window, then speeds up | `python game.py --progressive --player alice --log-trials data/alice_prog_1.csv` |

A measured or progressive session takes one to two minutes. When "EXPERIMENT COMPLETE" appears, press **Q** to quit.

## Controls

| Input | Action |
|---|---|
| **SPACE** | Start / pause / resume / retry |
| **Open hand** (or **thumb up** with `--gesture thumb`) | Jump |
| **D** | Debug overlay (shows the detected gesture, its score, and FPS) |
| **1 / 2 / 3** | Easy / normal / hard preset (normal mode only) |
| **+ / -** | Raise / lower the confidence threshold |
| **Q** or **Esc** | Quit |

## See your results

```bash
python analyze_trials.py data/alice_1.csv                      # plain-language report in the terminal
python analyze_trials.py data/alice_1.csv --html report.html   # charts, in your browser
python analyze_trials.py data/alice_1.csv --plot report.png    # four-panel picture
python analyze_trials.py data/alice_1.csv data/alice_2.csv     # pool several sessions
```

The key numbers are **bias** (do you jump early or late?) and **spread** (how scattered are your jumps?). The HTML report also checks whether your bias drifted during the session. [EXPERIMENT.md](EXPERIMENT.md) explains how to read them. About 30 jumps give wide uncertainty; pool sessions for firmer answers.

## Collecting good data

- **One file per session**, with a new name each time. Reusing a name overwrites the old file.
- **Use sessions from commit `0901def` or later.** Earlier versions tied the physics to the camera speed, so narrow windows were not what the log said.
- For comparing players, give everyone the same `--experiment-seed`. For repeat sessions of one player, change the seed so the order cannot be learned.
- A 30 FPS webcam is fine: physics runs at a fixed 1/60 s. The camera only affects when a gesture is noticed.
- CSV files are ignored by git (see `.gitignore`); add a `!data/name.csv` line to commit one on purpose.

## If something goes wrong

- **Window not visible:** it may open behind other windows; click the game icon in the dock or taskbar.
- **Camera unavailable:** close other apps using it, check OS permissions, try `--camera 1`.
- **No hand detected:** improve lighting, show the whole hand, move closer. The debug overlay (D) shows what the game sees.
- **Thumb not recognised:** keep the other fingers curled and the hand upright, and lift the thumb clearly above the knuckle. The threshold is `THUMB_RISE_MIN` in `gesture_lib.py`. The thumb option is new and tuned on idealized hands, so expect to adjust it.
- **Slow laptop:** lower the camera resolution or use `--preset 1`.

## How it fits together

| Layer | Files |
|---|---|
| Perception: camera frames to a jump event | `gesture_lib.py`, `mp_compat.py` |
| Game: physics, obstacles, difficulty | `game.py`, `adaptive.py`, `jump_service.py` |
| Experiment: planned obstacle lists, jump-time recording | `experiment.py`, `progressive.py` |
| Analysis | `analyze_trials.py`, `report_html.py`, `analyze.py`, `summarize.py` |

More: [EXPERIMENT.md](EXPERIMENT.md) (what is measured), [FUTURE.md](FUTURE.md) (ideas and research context), `AGENTS.md` (rules for contributors and agents), `gesture_landmarks_lab.ipynb` (how landmarks become features).

## Classic adaptive mode

Without `--experiment` or `--progressive`, a controller adapts a single difficulty number `d` to your success rate. It is fun for a few runs but settles into a rhythm, which is why the experiment modes exist.

```bash
python game.py --log-outcomes data/alice.csv --log-latency data/alice_latency.csv --player alice --max-crashes 0
python analyze.py data/alice.csv
python summarize.py data/alice.csv
```

After `--max-crashes N` (default 3) the game stops and shows stats; `0` disables that. See all options with `python game.py --help`.

## Difficulty presets

| Preset | d | Stability | Threshold |
|---|---|---|---|
| easy | 0.80 | 1 | 0.50 |
| normal | 0.94 | 2 | 0.60 |
| hard | 1.00 | 3 | 0.70 |

Presets only set the starting difficulty (`d`, confidence threshold, stability frames); the controller then adapts to your success rate. Stability is how many consecutive frames the gesture must be held before it counts as a jump.

## For developers

```bash
pytest -q                          # pure logic tests; camera tests skip without a webcam
python simulate_experiment.py --mu 0.03 --sigma 0.05 --out sim_trials.csv   # no camera needed
python dev_bench.py report         # difficulty and feasibility benchmarks
SDL_VIDEODRIVER=dummy python game.py --dummy-camera --smoke-frames 120      # headless smoke run
```

- Python 3.10 to 3.13. macOS, Windows and Linux should work; developed on macOS.
- Some MediaPipe wheels (Python 3.13, Apple Silicon) ship only the Tasks API; `mp_compat.py` falls back automatically.
- Tests: `tests/test_gesture_lib.py` (landmarks, classifiers), `tests/test_adaptive.py` (difficulty, physics, game loop), `tests/test_experiment.py`, `tests/test_progressive.py`, `tests/test_camera.py` (skipped without a camera).
