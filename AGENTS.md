# AGENTS.md

Slow-changing reference for the gesture lab. Regenerable numbers live in code, tests, and `dev_bench.py report`.

## File map
- `adaptive.py`: difficulty, physics, controller, CSV headers, `Phase`/`next_phase`.
- `game.py`: UI, camera, event loop; drives `next_phase`.
- `mp_compat.py`: MediaPipe compatibility shim.
- `dev_bench.py`: headless experiments.
- `analyze.py` / `summarize.py`: CSV reporting.

## Invariants
- `adaptive.py` and `gesture_lib.py` import no pygame, cv2, or mediapipe.
- `resolve_obstacles` is the only place the controller updates `d`.
- `Config` is the single tunable source; geometry and physics come from it.
- `adaptive.OUTCOME_HEADER` and `adaptive.LATENCY_HEADER` define CSV column order.
- `PRESETS` lives in `adaptive.py` and is re-exported by `game.py`.
- Phase flow lives in `adaptive.next_phase`; `game.py` only drives it.
- Smoke mode starts the game in `Phase.PLAYING`.
- The game loop stops on total crashes when `--max-crashes > 0`.
- `t_s` in outcome logs is game time (`state.play_time`), not wall-clock.
- Jitter parameters live in `Config`.

## Commands
```bash
pytest -q -k 'not test_frame_is_not_all_black'
SDL_VIDEODRIVER=dummy python game.py --dummy-camera --smoke-frames 120
python dev_bench.py feasibility
python dev_bench.py report
```

## Dependencies
- Core: `mediapipe`, `opencv-python`, `pygame`, `numpy`, `matplotlib`, `pytest`
- Notebook: `jupyter`

## Gotchas
- `mp_compat.py` prefers MediaPipe legacy `solutions`; when those are missing it falls back to the Tasks API using a downloaded `hand_landmarker.task` model. Pre-download it to `~/.cache/fontys_gesture_lab/hand_landmarker.task` for offline use.
- `game.py` and a few game-loop tests need pygame and cv2; pure tests never do.
