"""A gesture-controlled runner game with adaptive difficulty for the Fontys gesture lab.

Controls:
    SPACE      : start / pause / resume / retry
    Open hand  : jump (stable gesture, only while playing)
    1 / 2 / 3  : easy / normal / hard difficulty presets
    + / -      : raise / lower confidence threshold
    D          : toggle debug overlay
    Q / Esc    : quit
"""
import argparse
import csv
import dataclasses
import datetime
import random
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np
import pygame

from adaptive import (
    AdaptiveController,
    Config,
    LATENCY_HEADER,
    OUTCOME_HEADER,
    LatencyTracker,
    PRESETS,
    Phase,
    SessionTracker,
    attempt_jump,
    build_presets,
    difficulty_profile,
    next_phase,
    next_spawn_timer,
    resolve_obstacles,
    try_spawn,
)
from gesture_lib import RuleClassifier

WIDTH, HEIGHT = 1180, 720
CAM_W, CAM_H = 480, 360
FPS_TARGET = 60



def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def _write_csv_meta(file, config, extra=None):
    """Write config metadata as CSV comment lines before the header."""
    meta = dict(extra or {})
    meta["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["git_hash"] = _git_hash()
    meta["command"] = " ".join(sys.argv)
    for key, value in dataclasses.asdict(config).items():
        meta.setdefault(key, value)
    for key, value in meta.items():
        file.write(f"# {key}={value}\n")


class GameState:
    """Encapsulates the mutable game/session state so it can be unit-tested."""

    def __init__(self, controller, config, presets=None, start_preset=2):
        self.controller = controller
        self.config = config
        self.presets = presets if presets is not None else PRESETS
        self.ground_y = config.GROUND_Y
        self.player_x = config.PLAYER_X
        self.player_w = config.PLAYER_WIDTH
        self.player_h = config.PLAYER_HEIGHT
        self.gravity = config.GRAVITY
        self.jump_velocity = config.JUMP_VELOCITY
        self.tracker = SessionTracker()
        self.phase = Phase.READY
        self.play_time = 0.0
        self.preset_id = start_preset
        self.threshold = self.presets[self.preset_id]["threshold"]
        self.stability = self.presets[self.preset_id]["stability"]
        self.diff_color = self.presets[self.preset_id]["color"]
        self.player_y = self.ground_y - self.player_h
        self.velocity_y = 0.0
        self.obstacles = []
        self.spawn_timer = 0.8
        self.jump_latched = False
        self.stable_count = 0
        self.debug_overlay = False
        # Initialize controller and start first session.
        self.set_preset(self.preset_id)

    def set_preset(self, pid):
        self.controller.reset()
        self.preset_id = pid
        settings = self.presets[pid]
        self.controller.d = settings["d"]
        self.threshold = settings["threshold"]
        self.stability = settings["stability"]
        self.diff_color = settings["color"]
        self.tracker.bump()
        self.obstacles = []
        self.spawn_timer = 0.8
        self.last_spawn_t = None
        self.play_time = 0.0

    def reset_game(self):
        self.player_y = self.ground_y - self.player_h
        self.velocity_y = 0.0
        self.jump_latched = False
        self.stable_count = 0
        self.set_preset(self.preset_id)



class DummyCapture:
    """Stub camera for headless smoke runs."""

    def __init__(self, *_args, **_kwargs):
        self._frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def isOpened(self):
        return True

    def set(self, *_args, **_kwargs):
        return True

    def read(self):
        return True, self._frame.copy()

    def release(self):
        pass


def _float_in_closed_range(lo, hi, name):
    def f(value):
        v = float(value)
        if not lo <= v <= hi:
            raise argparse.ArgumentTypeError(
                f"{name} must be between {lo} and {hi}, got {v}"
            )
        return v
    return f


def _float_in_open_range(lo, hi, name):
    def f(value):
        v = float(value)
        if not lo < v < hi:
            raise argparse.ArgumentTypeError(
                f"{name} must be strictly between {lo} and {hi}, got {v}"
            )
        return v
    return f


def _positive_float_upto_one(name):
    def f(value):
        v = float(value)
        if not 0.0 < v <= 1.0:
            raise argparse.ArgumentTypeError(
                f"{name} must be > 0 and <= 1, got {v}"
            )
        return v
    return f


def _float_in_half_open_range(lo, hi, name):
    def f(value):
        v = float(value)
        if not lo <= v < hi:
            raise argparse.ArgumentTypeError(
                f"{name} must be >= {lo} and < {hi}, got {v}"
            )
        return v
    return f


def parse_args():
    parser = argparse.ArgumentParser(description="Gesture-controlled runner")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (0 on most Macs/Windows, sometimes 1)",
    )
    parser.add_argument(
        "--preset",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="Starting difficulty preset (default 2)",
    )
    parser.add_argument(
        "--start-d",
        type=_float_in_closed_range(0.0, 1.0, "--start-d"),
        default=None,
        help="Override the initial d for the starting preset (0-1)",
    )
    parser.add_argument(
        "--adaptation-rate",
        type=_positive_float_upto_one("--adaptation-rate"),
        default=Config.ADAPTATION_RATE,
        help="Controller adaptation rate / learning rate (default 0.08)",
    )
    parser.add_argument(
        "--momentum",
        type=_float_in_half_open_range(0.0, 1.0, "--momentum"),
        default=Config.MOMENTUM,
        help="Controller momentum, independent of the learning rate "
             "(0 <= m < 1, default 0 = off)",
    )
    parser.add_argument(
        "--target-success-rate",
        type=_float_in_open_range(0.0, 1.0, "--target-success-rate"),
        default=Config.TARGET_SUCCESS_RATE,
        help="Target success rate for the controller (default 0.75)",
    )
    parser.add_argument(
        "--spawn-mode",
        choices=("wild", "jitter", "probe"),
        default=Config.SPAWN_GAP_MODE,
        help="Obstacle rhythm: 'wild' = unpredictable gaps (default), "
             "'jitter' = near-constant gaps, 'probe' = measurement mode with "
             "very tight gaps set by --rest-range",
    )
    parser.add_argument(
        "--rest-range",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=None,
        help="Probe mode: rest time in seconds between landing and the next "
             "ideal take-off, sampled uniformly in [LO, HI] "
             "(default 0.0 0.7; the normal game never goes below ~0.44)",
    )
    parser.add_argument(
        "--spawn-room",
        type=_float_in_closed_range(0.0, 5.0, "--spawn-room"),
        default=Config.SPAWN_WILD_ROOM_S,
        help="Wild mode: average seconds of room above the shortest feasible "
             "gap (default 0.7; bigger = wilder and fewer obstacles at high d)",
    )
    parser.add_argument(
        "--dummy-camera",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--smoke-frames",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-crashes",
        type=int,
        default=3,
        help="Stop after this many total crashes (not consecutive); 0 disables.",
    )
    parser.add_argument(
        "--log-latency",
        type=str,
        default=None,
        help="Optional CSV path for jump-latency logs",
    )
    parser.add_argument(
        "--player",
        type=str,
        default="anon",
        help="Player name written to --log-outcomes",
    )
    parser.add_argument(
        "--log-outcomes",
        type=str,
        default=None,
        help="Optional CSV path for obstacle-outcome logs",
    )
    args = parser.parse_args()
    if args.rest_range is not None:
        lo, hi = args.rest_range
        if not (-0.3 <= lo <= hi <= 3.0):
            parser.error("--rest-range needs -0.3 <= LO <= HI <= 3.0")
    return args


def to_pygame_surface(bgr_frame):
    """Convert an OpenCV BGR frame to a Pygame RGB surface."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB")


def _draw_centered_panel(screen, font, small, lines, alpha=220):
    """Draw a translucent panel with centered text."""
    line_h = font.get_height() + 8
    small_h = small.get_height() + 4
    width = max(font.size(lines[0])[0] + 60, 360)
    height = line_h + (len(lines) - 1) * small_h + 40
    x = (WIDTH - width) // 2
    y = (HEIGHT - height) // 2
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    surf.fill((10, 14, 20, alpha))
    pygame.draw.rect(surf, (70, 85, 105, alpha), (0, 0, width, height), 2, border_radius=10)
    screen.blit(surf, (x, y))
    ly = y + 16
    for i, line in enumerate(lines):
        use_small = i != 0
        txt_surf = (small if use_small else font).render(line, True, (235, 240, 245))
        screen.blit(txt_surf, (x + (width - txt_surf.get_width()) // 2, ly))
        ly += small_h if use_small else line_h


def main():
    args = parse_args()

    pygame.init()
    pygame.display.set_caption("Fontys Gesture Lab | Open hand = jump")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    small = pygame.font.SysFont("consolas", 16)

    if args.dummy_camera:
        cap = DummyCapture(args.camera)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        pygame.quit()
        raise RuntimeError(
            f"Could not open camera at index {args.camera}. "
            "Try another index with --camera N."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    from mp_compat import mp_hands, mp_draw

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    classifier = RuleClassifier()

    # Adaptive-difficulty state
    config = Config(
        ADAPTATION_RATE=args.adaptation_rate,
        MOMENTUM=args.momentum,
        TARGET_SUCCESS_RATE=args.target_success_rate,
        D_START=args.start_d if args.start_d is not None else Config.D_START,
        SPAWN_GAP_MODE=args.spawn_mode,
        SPAWN_WILD_ROOM_S=args.spawn_room,
        SPAWN_REST_LO_S=(args.rest_range[0] if args.rest_range
                         else Config.SPAWN_REST_LO_S),
        SPAWN_REST_HI_S=(args.rest_range[1] if args.rest_range
                         else Config.SPAWN_REST_HI_S),
    )
    rng = random.Random()
    controller = AdaptiveController(config, rng=rng)
    presets = build_presets(args.start_d, args.preset)
    state = GameState(controller, config, presets=presets, start_preset=args.preset)

    # Latency diagnostics (rolling windows + counters)
    frame_to_jump = LatencyTracker(config.LATENCY_WINDOW)
    open_to_jump = LatencyTracker(config.LATENCY_WINDOW)
    infer_ms_tracker = LatencyTracker(config.LATENCY_WINDOW)
    loop_ms_tracker = LatencyTracker(config.LATENCY_WINDOW)
    latency_log = None
    if args.log_latency:
        latency_log = open(args.log_latency, "w", newline="")
        _write_csv_meta(
            latency_log, config,
            extra={"log_type": "latency", "player": args.player},
        )
        latency_writer = csv.writer(latency_log)
        latency_writer.writerow(
            LATENCY_HEADER
        )
    else:
        latency_writer = None

    outcomes_log = None
    outcomes_writer = None
    if args.log_outcomes:
        outcomes_log = open(args.log_outcomes, "w", newline="")
        _write_csv_meta(
            outcomes_log, config,
            extra={
                "log_type": "outcomes",
                "player": args.player,
                "initial_preset": args.preset,
                "effective_start_d": f"{controller.d:.4f}",
            },
        )
        outcomes_writer = csv.writer(outcomes_log)
        outcomes_writer.writerow(
            OUTCOME_HEADER
        )

    valid_gestures = 0
    accepted_jumps = 0
    rejected_jumps = 0

    detected = "none"
    score = 0.0
    state.stable_count = 0
    state.jump_latched = False
    first_open_time = 0.0
    camera_surface = None

    # Game state (constants kept local; mutable state lives in ``state``)
    ground_y = state.ground_y
    player_x = state.player_x
    player_w, player_h = state.player_w, state.player_h
    gravity = state.gravity
    jump_velocity = state.jump_velocity
    max_crashes = 0 if args.smoke_frames is not None else args.max_crashes
    d_history: list[float] = []

    over_printed = False

    def clear_session_counters():
        nonlocal valid_gestures, accepted_jumps, rejected_jumps, over_printed
        d_history.clear()
        valid_gestures = 0
        accepted_jumps = 0
        rejected_jumps = 0
        over_printed = False

    def draw_text(text, x, y, color=(235, 240, 245), use_small=False):
        surf = (small if use_small else font).render(text, True, color)
        screen.blit(surf, (x, y))

    def game_over_stats_lines():
        lines = [
            "GAME OVER",
            f"Crashed: {controller.failures}   Cleared: {controller.successes}",
            f"Final d: {controller.d:.3f}",
        ]
        if d_history:
            d_min = min(d_history)
            d_mean = statistics.mean(d_history)
            d_max = max(d_history)
            lines.append(f"d min / mean / max: {d_min:.3f} / {d_mean:.3f} / {d_max:.3f}")
        else:
            lines.append("d stats: (no obstacle outcomes recorded)")
        lines.append("Press SPACE to retry, Q to quit")
        return lines

    def on_outcome(ob, outcome):
        if outcomes_writer is not None:
            t_s = state.play_time
            outcomes_writer.writerow([
                f"{t_s:.3f}",
                args.player,
                state.tracker.current,
                state.preset_id,
                f"{ob.d_spawn:.4f}",
                f"{controller.d:.4f}",
                f"{ob.speed_spawn:.1f}",
                f"{ob.h:.1f}",
                f"{ob.w:.1f}",
                outcome,
                f"{ob.gap_s:.3f}",
            ])
            outcomes_log.flush()
        d_history.append(controller.d)

    if args.smoke_frames is not None:
        state.phase = Phase.PLAYING

    print("Game window is open (it may be behind other windows). Click it and press SPACE to start.")

    running = True
    while running:
        loop_start = time.perf_counter()
        raw_dt = min(clock.tick(FPS_TARGET) / 1000.0, 0.05)
        dt = raw_dt if state.phase == Phase.PLAYING else 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    new_phase = next_phase(state.phase, "space")
                    if state.phase == Phase.OVER and new_phase == Phase.PLAYING:
                        state.reset_game()
                        clear_session_counters()
                    state.phase = new_phase
                elif event.key == pygame.K_1:
                    state.set_preset(1)
                    state.phase = next_phase(state.phase, "preset")
                    clear_session_counters()
                elif event.key == pygame.K_2:
                    state.set_preset(2)
                    state.phase = next_phase(state.phase, "preset")
                    clear_session_counters()
                elif event.key == pygame.K_3:
                    state.set_preset(3)
                    state.phase = next_phase(state.phase, "preset")
                    clear_session_counters()
                elif event.key == pygame.K_d:
                    state.debug_overlay = not state.debug_overlay
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    state.threshold = min(0.95, state.threshold + 0.05)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    state.threshold = max(0.10, state.threshold - 0.05)

        ok, frame = cap.read()
        frame_read_time = time.perf_counter()
        infer_ms = 0.0
        if ok:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            infer_start = time.perf_counter()
            result = hands.process(rgb)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0

            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]
                detected, score = classifier.classify(landmarks)
                mp_draw.draw_landmarks(
                    frame,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2),
                )
            else:
                detected = "none"
                score = 0.0

            camera_surface = to_pygame_surface(
                cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
            )

        open_event = False
        if state.phase == Phase.PLAYING and detected == "open" and score >= state.threshold:
            if state.stable_count == 0:
                first_open_time = time.perf_counter()
            state.stable_count += 1
            if state.stable_count >= state.stability and not state.jump_latched:
                open_event = True
        else:
            state.stable_count = 0
            state.jump_latched = False

        if open_event:
            jump_time = time.perf_counter()
            state.velocity_y, accepted = attempt_jump(
                state.player_y, state.velocity_y, ground_y, player_h, jump_velocity
            )
            state.jump_latched = True
            valid_gestures += 1
            if accepted:
                accepted_jumps += 1
            else:
                rejected_jumps += 1

            frame_to_jump.add((jump_time - frame_read_time) * 1000.0)
            open_to_jump.add((jump_time - first_open_time) * 1000.0)
            infer_ms_tracker.add(infer_ms)

        # Player physics and world updates only while playing.
        speed = 0.0
        profile = difficulty_profile(controller.d, config)
        if state.phase == Phase.PLAYING:
            state.velocity_y += gravity * dt
            state.player_y += state.velocity_y * dt
            if state.player_y > ground_y - player_h:
                state.player_y = ground_y - player_h

            speed = profile.speed

            state.spawn_timer -= dt
            if state.spawn_timer <= 0.0:
                prev_ob = state.obstacles[-1] if state.obstacles else None
                new_ob = try_spawn(prev_ob, profile, speed, rng, config, controller.d)
                if new_ob is not None:
                    new_ob.gap_s = (
                        state.play_time - state.last_spawn_t
                        if state.last_spawn_t is not None else 0.0
                    )
                    state.last_spawn_t = state.play_time
                    state.obstacles.append(new_ob)
                    state.spawn_timer = next_spawn_timer(profile, rng, config)
                else:
                    # Retry on the next frame.
                    state.spawn_timer = min(dt, 0.01)

            state.obstacles = resolve_obstacles(
                state.obstacles,
                player_x,
                state.player_y,
                player_w,
                player_h,
                ground_y,
                speed,
                dt,
                controller,
                config,
                on_outcome=on_outcome,
            )

            if max_crashes > 0 and controller.failures >= max_crashes:
                state.phase = next_phase(state.phase, "game_over")

            state.play_time += dt
        else:
            speed = profile.speed

        loop_ms = (time.perf_counter() - loop_start) * 1000.0
        loop_ms_tracker.add(loop_ms)

        if open_event and latency_writer is not None:
            latency_writer.writerow(
                [
                    f"{frame_to_jump.values[-1]:.3f}",
                    f"{open_to_jump.values[-1]:.3f}",
                    f"{infer_ms:.3f}",
                    f"{loop_ms:.3f}",
                    int(accepted),
                ]
            )

        if args.smoke_frames is not None:
            args.smoke_frames -= 1
            if args.smoke_frames <= 0:
                running = False

        fps = clock.get_fps()

        screen.fill((18, 23, 33))
        pygame.draw.rect(screen, (28, 36, 49), (0, 0, WIDTH, 385))

        if camera_surface is not None:
            screen.blit(camera_surface, (24, 24))
        else:
            pygame.draw.rect(screen, (35, 42, 55), (24, 24, CAM_W, CAM_H))
            draw_text("Camera preview unavailable", 40, 45)

        pygame.draw.rect(screen, (75, 91, 112), (20, 20, CAM_W + 8, CAM_H + 8), 2)

        draw_text(f"Difficulty preset: {state.preset_id} (d={controller.d:.2f})", 525, 25)
        pygame.draw.rect(screen, state.diff_color, (780, 28, 20, 14), border_radius=3)
        draw_text(f"Gesture: {detected} ({score:.2f})", 525, 58)
        draw_text(f"Threshold: {state.threshold:.2f}  (+ / -)", 525, 91)
        draw_text(f"Speed: {speed:.0f}", 525, 124)
        draw_text(f"FPS: {fps:.1f}  |  Loop: {loop_ms:.1f} ms", 525, 157)
        draw_text(
            "Controls: SPACE start/pause/resume/retry  1/2/3 presets  +/- threshold  D debug  Q quit",
            525,
            190,
            use_small=True,
        )
        pygame.draw.rect(screen, (13, 17, 24), (0, 390, WIDTH, HEIGHT - 390))
        pygame.draw.line(screen, (110, 125, 145), (0, ground_y), (WIDTH, ground_y), 3)
        pygame.draw.rect(
            screen,
            (88, 197, 255),
            (int(player_x), int(state.player_y), player_w, player_h),
            border_radius=7,
        )
        for ob in state.obstacles:
            color = (120, 120, 120) if ob.failed else (255, 111, 105)
            rect = pygame.Rect(
                int(ob.x), int(ground_y - ob.h), int(ob.w), int(ob.h)
            )
            pygame.draw.rect(screen, color, rect, border_radius=5)

        draw_text(f"Cleared: {controller.successes}  Crashed: {controller.failures}", 24, 410)
        draw_text("Open hand = jump. Close hand to re-arm.", 24, 445, use_small=True)

        if state.debug_overlay:
            recent = controller.recent_success_rate()
            lines = [
                "--- debug ---",
                f"d = {controller.d:.3f}",
                f"speed={speed:.1f} h={profile.height:.1f} w={profile.width:.1f} int={profile.spawn_interval:.2f}",
                f"recent success = {recent:.2f} (last {len(controller.recent)})",
                f"totals: {controller.outcomes}  ok={controller.successes}  bad={controller.failures}",
                f"gestures: valid={valid_gestures} acc={accepted_jumps} rej={rejected_jumps}",
            ]
            if frame_to_jump.count >= 10:
                lines.append(
                    f"latency ms: mean={frame_to_jump.mean():.1f} "
                    f"med={frame_to_jump.median():.1f} "
                    f"p90={frame_to_jump.p90():.1f} "
                    f"std={frame_to_jump.std():.1f}"
                )
            if loop_ms_tracker.count >= 10:
                lines.append(
                    f"loop ms: mean={loop_ms_tracker.mean():.1f} "
                    f"p90={loop_ms_tracker.p90():.1f}"
                )
            y = 480
            for line in lines:
                draw_text(line, 525, y, use_small=True)
                y += 20

        if state.phase == Phase.READY:
            _draw_centered_panel(screen, font, small, ["Press SPACE to start", "Open hand = jump"])
        elif state.phase == Phase.PAUSED:
            _draw_centered_panel(screen, font, small, ["Paused", "Press SPACE to resume"])
        elif state.phase == Phase.OVER:
            if not over_printed:
                print("\n" + "=" * 40)
                for line in game_over_stats_lines():
                    print(line)
                print("=" * 40)
                over_printed = True
            _draw_centered_panel(screen, font, small, game_over_stats_lines(), alpha=230)

        pygame.display.flip()

    hands.close()
    cap.release()
    if latency_log is not None:
        latency_log.close()
    if outcomes_log is not None:
        outcomes_log.close()
    pygame.quit()


if __name__ == "__main__":
    main()
