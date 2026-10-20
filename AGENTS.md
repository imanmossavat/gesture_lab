# AGENTS.md

Slow-changing reference for the gesture lab. Regenerable numbers live in code, tests, and `dev_bench.py report`.

## File map
- `gesture_lab/adaptive.py`: difficulty, physics, controller, CSV headers, `Phase`/`next_phase`.
- `gesture_lab/game.py`: UI, camera, event loop; drives `next_phase`.
- `gesture_lab/mp_compat.py`: MediaPipe compatibility shim.
- `gesture_lab/dev_bench.py`: headless experiments.
- `gesture_lab/analyze.py` / `gesture_lab/summarize.py`: CSV reporting.
- `gesture_lab/jump_service.py`: pure physics: jump window `[L, U]`, window-to-height, timing models.
- `gesture_lab/experiment.py`: planned obstacle list (`make_trials`), jump-time recorder, `TRIAL_HEADER`.
- `gesture_lab/simulate_experiment.py`: headless simulated player that writes a trial log.
- `gesture_lab/analyze_trials.py`: plain-language report and plot for trial logs. See `docs/EXPERIMENT.md`.
- `gesture_lab/report_html.py`: one-file HTML report (charts + drift model); `analyze_trials.py --html` calls it.
- `gesture_lab/progressive.py`: `ProgressivePlan` (speed ladder + staircase); same interface as `ExperimentPlan` plus `report()`.
- `gesture_lab/gesture_lib.py`: also `ThumbClassifier` / `make_classifier` for `game.py --gesture open|thumb`.
- `docs/FUTURE.md`: research context and ideas not built yet.

## Invariants
- `gesture_lab/adaptive.py`, `gesture_lab/gesture_lib.py`, `gesture_lab/jump_service.py` and `gesture_lab/experiment.py` import no pygame, cv2, or mediapipe.
- `experiment.TRIAL_HEADER` defines the trial-log column order (separate from `OUTCOME_HEADER`).
- Experiment windows use `experiment.game_window`, not the textbook `jump_service.success_window` (frame-step corrections).
- `resolve_obstacles` is the only place the controller updates `d`.
- `Config` is the single tunable source; geometry and physics come from it.
- `adaptive.OUTCOME_HEADER` and `adaptive.LATENCY_HEADER` define CSV column order.
- `PRESETS` lives in `gesture_lab/adaptive.py` and is re-exported by `gesture_lab/game.py`.
- Phase flow lives in `adaptive.next_phase`; `gesture_lab/game.py` only drives it.
- Smoke mode starts the game in `Phase.PLAYING`.
- The game loop stops on total crashes when `--max-crashes > 0`.
- `t_s` in outcome logs is game time (`state.play_time`), not wall-clock.
- Jitter parameters live in `Config`.

## Commands
```bash
pytest -q -k 'not test_frame_is_not_all_black'
SDL_VIDEODRIVER=dummy python game.py --dummy-camera --smoke-frames 120
python -m gesture_lab.dev_bench feasibility
python -m gesture_lab.dev_bench report
```

## Dependencies
- Core: `mediapipe`, `opencv-python`, `pygame`, `numpy`, `matplotlib`, `pytest`
- Notebook: `jupyter`

## Gotchas
- `gesture_lab/mp_compat.py` prefers MediaPipe legacy `solutions`; when those are missing it falls back to the Tasks API using a downloaded `hand_landmarker.task` model. Pre-download it to `~/.cache/fontys_gesture_lab/hand_landmarker.task` for offline use.
- `gesture_lab/game.py` and a few game-loop tests need pygame and cv2; pure tests never do.

## Timing (grep `TIMING:` for the matching code notes)
- Physics runs in fixed `experiment.FRAME_DT` (1/60 s) steps. Why: gravity makes the jump arc depend on the step size, and the experiment windows are computed for that step. The game loop is camera-bound (`cap.read()` blocks), so frame time is the camera interval, not `FPS_TARGET`. Never use it as the physics step; `adaptive.fixed_steps` converts it to whole steps.
- Camera speed only changes when a gesture is noticed (up to one camera interval, plus `stability` frames). It never changes windows or outcomes.
- Two clocks in the trial log. `t_j`/`delta`: when the game started the jump; decides the outcome, so the physics check and success model use it. `t_open`/`delta_open`: first camera frame that showed the gesture minus half a camera interval; bias, spread and drift use it. Do not mix them.
- Not correctable in software, so it stays in `delta_open`: the camera's internal delay and the player's movement time. Compare spreads, not biases, across gestures or setups.
- `TRIAL_HEADER` is append-only.
