# Gesture Lab

A hand-controlled runner that doubles as a small experiment: how precisely can you time a jump with a hand gesture? Play it, then get a report on your own timing. Needs a webcam and Python 3.10 to 3.13.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python game.py
```

A window opens (it may hide behind other windows). Click it and press **SPACE**.

- **Setup:** sit about an arm's length from the camera, whole hand in view, fingers up, good light.
- **Jump:** open your hand, then **close it** before the next jump (that re-arms it).
- **First run needs internet once** on installs that download the hand model. Offline: put `hand_landmarker.task` in `~/.cache/fontys_gesture_lab/`.

## Modes

| I want to... | Command |
|---|---|
| Just play | `python game.py` |
| Jump with a thumb instead of an open hand | `python game.py --gesture thumb` (thumb up from a fist; lower it to re-arm) |
| Be measured: a fixed obstacle list, identical for the same seed | `python game.py --experiment --experiment-seed 42 --player alice --log-trials data/alice_1.csv` |
| A progressive challenge: it finds your narrowest feasible window, then speeds up | `python game.py --progressive --player alice --log-trials data/alice_prog_1.csv` |

A measured or progressive session takes one to two minutes. When "EXPERIMENT COMPLETE" appears, press **Q**. All options: `python game.py --help`.

## Controls

| Input | Action |
|---|---|
| **SPACE** | Start / pause / resume / retry |
| **Open hand** (**thumb up** with `--gesture thumb`) | Jump |
| **D** | Debug overlay: detected gesture, score, FPS |
| **1 / 2 / 3** | Easy / normal / hard preset (sets confidence threshold and stability frames, and restarts the run) |
| **+ / -** | Raise / lower the confidence threshold |
| **Q** or **Esc** | Quit |

## Your results

```bash
python analyze_trials.py data/alice_1.csv                      # plain-language report
python analyze_trials.py data/alice_1.csv --html report.html   # charts, with a drift check
python analyze_trials.py data/alice_1.csv --plot report.png    # four-panel picture
python analyze_trials.py data/alice_1.csv data/alice_2.csv     # pool sessions
```

The key numbers are **bias** (early or late?) and **spread** (how scattered?). Both are measured from when the camera first *saw* your gesture, with the game's own wait removed; [EXPERIMENT.md](EXPERIMENT.md) explains the two clocks and how to read the report. About 30 jumps give wide uncertainty, so pool sessions for firmer answers.

## Collecting good data

- **One file per session**, with a new name each time. Reusing a name overwrites the old file.
- **Same seed to compare players** (`--experiment-seed`); a new seed for repeat sessions of one player, so the order cannot be learned.
- **Same preset for everyone** (`--preset`). It sets how many frames a gesture must be held, which adds delay.
- Camera speed (a 30 FPS webcam is fine) only changes when a gesture is noticed, never the windows: physics runs at a fixed 1/60 s. Compare biases only on the same setup (see `AGENTS.md`, Timing).
- Logs made before physics ran at a fixed 1/60 s are not comparable for tall obstacles; the header's `git_hash` says which code wrote a file.
- CSV files are ignored by git; add a `!data/name.csv` line to `.gitignore` to commit one on purpose.

## Troubleshooting

- **No window:** it may be behind others; click its icon in the dock or taskbar.
- **Camera unavailable:** close other apps using it, check OS permissions, try `--camera 1`.
- **No hand detected:** more light, whole hand in view, move closer. Press **D** to see what the game sees.
- **Thumb not recognised:** curl the other fingers, keep the hand upright, lift the thumb clearly above the knuckle. The threshold is `THUMB_RISE_MIN` in `gesture_lib.py`; it was tuned on idealized hands, so adjust it for yours.
- **Gestures feel late:** check FPS in the debug overlay; a slow camera delays recognition but does not change the physics.

## Project map

| Layer | Files |
|---|---|
| Perception: camera frames to a jump | `gesture_lib.py`, `mp_compat.py` |
| Game: physics, obstacles, difficulty | `game.py`, `adaptive.py`, `jump_service.py` |
| Experiment: obstacle plans, jump-time recording | `experiment.py`, `progressive.py` |
| Analysis | `analyze_trials.py`, `report_html.py`, `analyze.py`, `summarize.py` |

Docs: [EXPERIMENT.md](EXPERIMENT.md) (what is measured), [FUTURE.md](FUTURE.md) (ideas, research context), `AGENTS.md` (rules for contributors and agents), `gesture_landmarks_lab.ipynb` (how landmarks become features).

## Classic adaptive mode

Without `--experiment` or `--progressive`, a controller adapts one difficulty number `d` to your success rate. It is fun for a few runs but settles into a rhythm, which is why the experiment modes exist.

```bash
python game.py --log-outcomes data/alice.csv --log-latency data/alice_latency.csv --player alice --max-crashes 0
python analyze.py data/alice.csv
python summarize.py data/alice.csv
```

After `--max-crashes N` (default 3) the game stops and shows stats; `0` disables that.

## Difficulty presets

| Preset | d | Stability | Threshold |
|---|---|---|---|
| easy | 0.80 | 1 | 0.50 |
| normal | 0.94 | 2 | 0.60 |
| hard | 1.00 | 3 | 0.70 |

A preset sets the starting `d`, the confidence threshold, and the stability (consecutive frames a gesture must be held to count as a jump). Only `d` then adapts.

## Development

```bash
pytest -q                                                                  # camera tests skip without a webcam
python simulate_experiment.py --mu 0.03 --sigma 0.05 --out sim_trials.csv  # no camera needed
python dev_bench.py feasibility                                            # clearance checks (see --help for more)
SDL_VIDEODRIVER=dummy python game.py --dummy-camera --smoke-frames 120     # headless smoke run
```

Some MediaPipe wheels (for example Python 3.13 on Apple Silicon) ship only the Tasks API; `mp_compat.py` falls back automatically. Tests live in `tests/`.
