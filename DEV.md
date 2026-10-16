# Developer guide: adaptive difficulty

This document explains the physics-informed adaptive difficulty system and how
to tune it.

## Files

| File | Responsibility |
|---|---|
| `adaptive.py` | Pure-Python difficulty model, physics helpers, controller, obstacle resolution, latency tracker. |
| `mp_compat.py` | Compatibility shim for MediaPipe legacy `solutions`. Falls back to no-op stubs on wheels that only ship the Tasks API. |
| `game.py` | UI, camera, event loop. Imports all adaptive logic from `adaptive.py` and uses `mp_compat`. |
| `dev_bench.py` | Headless bench for tuning: feasibility, sweep, simulate, compare, latency-report. |
| `tests/test_adaptive.py` | Unit tests for the adaptive system. |

## Difficulty model

`d` is a normalized scalar in `[0, 1]`.

```text
speed = 260 + 170*d
height = 30 + 40*d
width  = 20 + 15*d
spawn_interval = 2.5 - 1.2*d
```

Height is jittered ±10% at spawn time.  Width is **not** jittered because the
timing window depends on the horizontal overlap `(w + PLAYER_WIDTH) / speed`,
and width jitter would break the min-gap guarantee.

The actual time between spawns is `spawn_interval * uniform(1.0, 1.3)`, which
only ever lengthens the base interval.

## Physics helpers

The jump parabola is derived from the constants measured in the existing game:

```text
GRAVITY       = 1800 px/s^2
JUMP_VELOCITY = -690 px/s
```

`time_above(h)` returns the duration during a jump when the player's feet are
at least `h` pixels above the ground.  `timing_window(h, w, speed)` subtracts
the horizontal overlap duration `(w + PLAYER_WIDTH) / speed` from
`time_above(h)`.

A perfect-timed jump clears the obstacle when `timing_window > 0`.  The usable
timing budget is half of that margin (`clears_obstacle`).

## Controller

```text
d <- clip(d + 0.08 * (outcome - 0.75))
```

- success (outcome = 1): `d += +0.02`
- failure (outcome = 0): `d += -0.06`

The target success rate is 75%.  The controller is intentionally asymmetric:
it recovers faster from being too hard than it drifts upward, which prevents
catastrophic death spirals.

`controller.recent` is a diagnostic window only; it does **not** drive the
update.

## Spawn control / min-gap

When the spawn timer expires, `try_spawn` checks the physical gap from the
previous obstacle's right edge to the new obstacle's left edge at spawn time.

```text
required_gap_px = max(350, speed * (AIR_TIME + 0.35))
```

- `AIR_TIME` is the total jump air time (~0.767 s).
- `0.35 s` is a landing margin before the next jump must be attempted.
- The 350 px floor prevents the rule from collapsing at low difficulty.

If the gap is too small, `try_spawn` returns `None` and the caller retries on
the next frame.  Off-screen obstacles are ignored for spacing.

## Outcome resolution

`resolve_obstacles` is the only place the controller is updated.  It records
exactly one outcome per obstacle:

- **Failure** on the first frame of rectangle overlap.
- **Success** when the obstacle's right edge passes the player's left edge
  without ever overlapping.
- `done` prevents double counting.
- Off-screen obstacles are removed.

## Latency tracking

On every accepted/rejected jump event the game records:

- `frame_read_to_jump_ms` — time from `cap.read()` returning to the jump being
  issued.
- `first_open_to_jump_ms` — time from the first frame of the current stable
  open-hand streak to the jump.
- `inference_ms` — MediaPipe hand-landmark inference time.
- `loop_ms` — full frame loop time.

Rolling-window statistics (mean, median, p90, std) are shown in the debug
overlay (press **D** in-game).

Use `--log-latency latency.csv` to write every jump event to disk.

## Outcome logging

Use `--log-outcomes path.csv` to write one row per obstacle outcome:

```text
t_s, player, session, mode, d_spawn, d_after, speed, height, width, outcome
```

- `t_s` is seconds since the last restart.
- `player` is set with `--player` (default `anon`).
- `session` is an integer that increments every time the player changes the
  difficulty preset (1/2/3) or presses **R** to restart.  Analyze treats each
  session as a separate run, so settling and tail statistics are computed per
  session.
- `mode` is the in-game difficulty preset (1/2/3).
- `d_spawn` is the controller `d` used to spawn the obstacle.
- `d_after` is `d` after the controller has processed the outcome.
- `outcome` is `1` for a successful clear and `0` for a crash.

Analyze the log with:

```bash
python analyze.py run1.csv
python analyze.py run1.csv --plot report.png
```

### Run metadata in CSV outputs

`--log-outcomes` and `--log-latency` write a metadata block as leading `# key=value`
comment lines before the CSV header.  The block contains:

- run context: `log_type`, `player`, `initial_preset`, `effective_start_d`,
  `timestamp`, `git_hash`, `command`
- every tunable field from the active `Config` (e.g. `ADAPTATION_RATE`,
  `MOMENTUM`, `TARGET_SUCCESS_RATE`, `D_START`, `RECENT_WINDOW`, `SPEED_MIN/MAX`, ...)

`analyze.py` and `summarize.py` read these comments and show the tuning
parameters with each report, so it is always clear which configuration produced
the numbers.

## Running checks

```bash
# Fast unit tests (no camera, no display)
pytest -q

# Headless smoke run
SDL_VIDEODRIVER=dummy python game.py --dummy-camera --smoke-frames 120

# Tune the difficulty model
python dev_bench.py feasibility
python dev_bench.py sweep
python dev_bench.py simulate --frames 2000 --noise 80 --latency 90 --csv simulate.csv
python dev_bench.py compare --output compare.png --csv compare.csv
python dev_bench.py latency-report --max-latency 200 --output latency.png --csv latency.csv

# Consume the resulting CSVs
python analyze.py simulate.csv
python summarize.py simulate.csv
```

## What is theoretical / not calibrated

Everything in `Config` is a design choice, not a measurement:

- Difficulty profile constants (`SPEED_MIN/MAX`, `HEIGHT_MIN/MAX`, ...).
- Jitter ranges.
- Controller `ETA` and `TARGET_SUCCESS_RATE`.
- Landing margin (`0.35 s`) and min-gap floor (`350 px`).

Only `GRAVITY`, `JUMP_VELOCITY`, and player geometry are measured from the
existing game.  Expect to adjust the theoretical values after play-tests.

## MediaPipe note

Recent MediaPipe wheels (especially Python 3.13 / Apple Silicon) ship the Tasks
API without the legacy `mediapipe.python.solutions` module.  `mp_compat.py`
gracefully degrades to a no-op hand detector so the game can still be imported
and smoke-tested.  For real hand tracking on those wheels, install a MediaPipe
build that includes legacy solutions, or port the game to the Tasks API with a
`hand_landmarker.task` model.
