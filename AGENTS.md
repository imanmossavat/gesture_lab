# AGENTS.md

This file is the slow-changing reference for anyone modifying the gesture lab.
Keep it short; regenerable numbers live in code, tests, and `dev_bench.py report`.

## File map

`adaptive.py` is the pure source of truth for difficulty, physics, controller, and CSV headers; `game.py` adds UI/camera; `dev_bench.py` runs headless experiments.

## Invariants

- `adaptive.py` and `gesture_lib.py` import no pygame, cv2, or mediapipe.
- `resolve_obstacles` is the only place the controller updates `d`.
- `Config` is the single tunable source; geometry and physics come from it, not hardcoded game constants.
- `adaptive.OUTCOME_HEADER` and `adaptive.LATENCY_HEADER` define CSV column order.
- `PRESETS` lives in `adaptive.py` and is re-exported by `game.py`.

## Commands

```bash
pytest -q -k 'not test_frame_is_not_all_black'
SDL_VIDEODRIVER=dummy python game.py --dummy-camera --smoke-frames 120
python dev_bench.py feasibility
python dev_bench.py report
```

## Gotchas

- `mp_compat.py` silently degrades to a no-op hand detector on MediaPipe wheels that lack legacy `solutions`; real tracking needs a legacy-solutions build or a Tasks API port.
- `game.py` and a few game-loop tests need pygame; pure tests never do.
