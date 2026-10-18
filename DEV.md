# Developer guide

## Files

| File | Responsibility |
|---|---|
| `adaptive.py` | Pure difficulty model, physics, controller, obstacle resolution, latency tracker, CSV headers, `Phase`/`next_phase`. |
| `mp_compat.py` | MediaPipe compatibility shim. |
| `game.py` | UI, camera, event loop; drives `next_phase`. |
| `dev_bench.py` | Headless bench for tuning. |
| `tests/test_adaptive.py` | Unit tests. |

## Phase flow

The game loop uses `adaptive.Phase` and `adaptive.next_phase`. READY/PAUSED/OVER frames use `dt = 0` and ignore jumps. SPACE advances the phase; preset keys always return to READY. Smoke mode forces PLAYING. See `game.py`.

## Stop rule

`--max-crashes N` ends a session after N total crashes (not consecutive). For short runs, binomial success rates are noisy; use confidence intervals (Wilson) rather than point estimates when judging target success.

## Jitter

Spawn interval and obstacle height are sampled from normal distributions defined in `Config`. Use `mean - 2σ` as the design-margin schedule floor and `mean + 2σ` for clearance. Width is not jittered because it would break the min-gap guarantee.

## Difficulty model

`d ∈ [0, 1]`. `difficulty_profile(d, cfg)` returns speed, height, width, and spawn interval. Run `python dev_bench.py report` for the current numbers.

## Physics

Jump parabola comes from `Config.GRAVITY` and `Config.JUMP_VELOCITY`. Use `air_time()`, `jump_peak()`, `apex_time()`, and `time_above(h)` to inspect it. `timing_window(h, w, speed)` returns the available timing margin; `clears_obstacle` uses half of that margin.

## Controller

`AdaptiveController.update` moves `d` toward `Config.TARGET_SUCCESS_RATE` using `ADAPTATION_RATE` and optional `MOMENTUM`. `controller.recent` is diagnostic only. The report prints per-outcome deltas.

## Spawn rhythm (`SPAWN_GAP_MODE`)

Default `wild`: each gap is drawn uniformly between the shortest feasible gap (`min_spawn_interval`, the min-gap rule in seconds) and `2*mean - min`, where `mean` is the profile interval but at least `min + SPAWN_WILD_ROOM_S` (0.7 s). Gaps are therefore never below the feasible minimum, never much longer than twice the mean, and unpredictable (sd about 0.4-0.6 s, well above a player's timing noise). At high `d` the profile interval is only about 0.1 s above the physical floor, so the 0.7 s room raises the average gap there (about 1.9 s at `d = 1`); fewer obstacles at the top, difficulty comes from speed and size. `jitter` restores the old near-metronome behaviour. Flags: `--spawn-mode {wild,jitter}`, `--spawn-room SECONDS`. Both are logged in the CSV header.

### Probe mode (tight gaps, for measuring recovery time)

`--spawn-mode probe --rest-range LO HI` samples the **rest time** directly: spawn-to-spawn gap = air time + U(LO, HI). Rest is the time between landing and the ideal take-off for the next obstacle when the previous jump was perfectly timed (negative rest is impossible without early/late luck). The normal game never goes below about 0.44 s of rest; probe mode ignores the profile interval and the min-gap floor (only overlap is forbidden), default range 0.0-0.7 s. Use a fixed difficulty for clean data: `--start-d 0.5 --adaptation-rate 0.000001`. Each outcome row now has a `gap_s` column (time since the previous spawn); `summarize.py` turns it into "success by rest time" (previous obstacle cleared; compared with the timing-only model fitted on long-rest pairs; bins marked `<< below model` show recovery/decision time, not timing precision, is the limit).

## Spawn / min-gap

`try_spawn` calls `spawn_allowed` to enforce a minimum physical gap (`MIN_GAP_FLOOR_PX`, `LANDING_MARGIN_S`, `air_time()`). If the gap is too small, the spawn is retried next frame.

## Outcome resolution

`resolve_obstacles` is the only `d` updater. It records exactly one outcome per obstacle: failure on first overlap, success when the obstacle passes safely, `done` prevents double counting, and off-screen obstacles are removed.

## Latency / outcome logging

- `--log-latency path.csv` writes one row per jump event.
- `--log-outcomes path.csv` writes one row per obstacle outcome; `t_s` is game time.
- Analyze with `analyze.py` and `summarize.py`.

## Running checks

```bash
pytest -q
SDL_VIDEODRIVER=dummy python game.py --dummy-camera --smoke-frames 120
python dev_bench.py feasibility
python dev_bench.py report
python dev_bench.py simulate --frames 2000 --csv sim.csv
python analyze.py sim.csv
```

## MediaPipe note

Recent MediaPipe wheels (especially Python 3.13 / Apple Silicon) ship only the Tasks API. `mp_compat.py` falls back automatically and downloads `hand_landmarker.task` to `~/.cache/fontys_gesture_lab/` on first use. Pre-download the model for offline use.

## Ideas not implemented

- Consecutive-crash stop rule instead of total crashes.
- Per-preset `Config` overrides.
- Optional neural hand detector backend.

## What is theoretical

Everything in `Config` is a design choice, not a measurement; expect to tune after play-tests. Only player geometry, `GRAVITY`, and `JUMP_VELOCITY` are measured from the existing game.
