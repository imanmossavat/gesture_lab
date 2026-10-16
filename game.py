"""A gesture-controlled runner game with adaptive difficulty for the Fontys gesture lab.

Controls:
    Open hand  : jump (stable gesture)
    1 / 2 / 3  : easy / normal / hard difficulty presets (sets d to 0.25/0.5/0.75)
    + / -      : raise / lower confidence threshold
    R          : restart
    D          : toggle debug overlay
    Q / Esc    : quit
"""
import argparse
import csv
import random
import time

import cv2
import numpy as np
import pygame

from adaptive import (
    AdaptiveController,
    Config,
    LatencyTracker,
    attempt_jump,
    difficulty_profile,
    resolve_obstacles,
    try_spawn,
)
from gesture_lib import RuleClassifier

WIDTH, HEIGHT = 1180, 720
CAM_W, CAM_H = 480, 360
FPS_TARGET = 60

# Presets only affect the starting d, the classifier threshold, the stability
# requirement, and the UI colour.  All runtime parameters come from
# difficulty_profile(d).
PRESETS = {
    1: {"d": 0.25, "threshold": 0.50, "stability": 1, "color": (120, 220, 120)},
    2: {"d": 0.50, "threshold": 0.60, "stability": 2, "color": (255, 200, 90)},
    3: {"d": 0.75, "threshold": 0.70, "stability": 3, "color": (255, 110, 110)},
}


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


def parse_args():
    parser = argparse.ArgumentParser(description="Gesture-controlled runner")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (0 on most Macs/Windows, sometimes 1)",
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
        "--log-latency",
        type=str,
        default=None,
        help="Optional CSV path for jump-latency logs",
    )
    return parser.parse_args()


def to_pygame_surface(bgr_frame):
    """Convert an OpenCV BGR frame to a Pygame RGB surface."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB")


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
    config = Config()
    rng = random.Random()
    controller = AdaptiveController(config, rng=rng)

    # Latency diagnostics (rolling windows + counters)
    frame_to_jump = LatencyTracker(config.LATENCY_WINDOW)
    open_to_jump = LatencyTracker(config.LATENCY_WINDOW)
    infer_ms_tracker = LatencyTracker(config.LATENCY_WINDOW)
    loop_ms_tracker = LatencyTracker(config.LATENCY_WINDOW)
    latency_log = None
    if args.log_latency:
        latency_log = open(args.log_latency, "w", newline="")
        latency_writer = csv.writer(latency_log)
        latency_writer.writerow(
            ["frame_read_to_jump_ms", "first_open_to_jump_ms", "inference_ms", "loop_ms", "accepted"]
        )
    else:
        latency_writer = None

    valid_gestures = 0
    accepted_jumps = 0
    rejected_jumps = 0

    detected = "none"
    score = 0.0
    stable_count = 0
    jump_latched = False
    first_open_time = 0.0
    camera_surface = None

    # Game state
    ground_y = 590
    player_x = 620
    player_w, player_h = 42, 56
    player_y = ground_y - player_h
    velocity_y = 0.0
    gravity = 1800.0
    jump_velocity = -690.0
    obstacles = []
    spawn_timer = 0.8
    debug_overlay = False

    running = True
    preset_id = 2
    threshold = PRESETS[preset_id]["threshold"]
    stability = PRESETS[preset_id]["stability"]
    diff_color = PRESETS[preset_id]["color"]

    def set_preset(pid):
        nonlocal preset_id, threshold, stability, diff_color
        preset_id = pid
        settings = PRESETS[pid]
        controller.d = settings["d"]
        threshold = settings["threshold"]
        stability = settings["stability"]
        diff_color = settings["color"]

    def reset_game():
        nonlocal player_y, velocity_y, obstacles, spawn_timer
        nonlocal jump_latched, stable_count, debug_overlay
        player_y = ground_y - player_h
        velocity_y = 0.0
        obstacles = []
        spawn_timer = 0.8
        jump_latched = False
        stable_count = 0
        controller.reset()
        set_preset(preset_id)

    def draw_text(text, x, y, color=(235, 240, 245), use_small=False):
        surf = (small if use_small else font).render(text, True, color)
        screen.blit(surf, (x, y))

    set_preset(preset_id)

    while running:
        loop_start = time.perf_counter()
        dt = min(clock.tick(FPS_TARGET) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    reset_game()
                elif event.key == pygame.K_1:
                    set_preset(1)
                elif event.key == pygame.K_2:
                    set_preset(2)
                elif event.key == pygame.K_3:
                    set_preset(3)
                elif event.key == pygame.K_d:
                    debug_overlay = not debug_overlay
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    threshold = min(0.95, threshold + 0.05)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    threshold = max(0.10, threshold - 0.05)

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
        if detected == "open" and score >= threshold:
            if stable_count == 0:
                first_open_time = time.perf_counter()
            stable_count += 1
            if stable_count >= stability and not jump_latched:
                open_event = True
        else:
            stable_count = 0
            jump_latched = False

        if open_event:
            jump_time = time.perf_counter()
            velocity_y, accepted = attempt_jump(
                player_y, velocity_y, ground_y, player_h, jump_velocity
            )
            jump_latched = True
            valid_gestures += 1
            if accepted:
                accepted_jumps += 1
            else:
                rejected_jumps += 1

            frame_to_jump.add((jump_time - frame_read_time) * 1000.0)
            open_to_jump.add((jump_time - first_open_time) * 1000.0)
            infer_ms_tracker.add(infer_ms)

            if latency_writer is not None:
                # loop_ms is not known yet; we write it in the next frame by
                # keeping the last jump's pending row.  Simpler: log without loop
                # and record loop_ms separately below.
                pass

        # Player physics (always runs; no game over)
        velocity_y += gravity * dt
        player_y += velocity_y * dt
        if player_y > ground_y - player_h:
            player_y = ground_y - player_h
            velocity_y = 0.0

        # Speed policy: one global speed from the current d.  The controller
        # changes d by at most ETA = 0.08 per outcome, so speed changes by at
        # most 170 * 0.08 = 13.6 px/s per step (in practice <= ~10 px/s near
        # the target).  This keeps the pixel gap between obstacles stable.
        profile = difficulty_profile(controller.d, config)
        speed = profile.speed

        # Spawn logic with min-gap retry.
        spawn_timer -= dt
        if spawn_timer <= 0.0:
            prev_ob = obstacles[-1] if obstacles else None
            new_ob = try_spawn(prev_ob, profile, speed, rng, config)
            if new_ob is not None:
                obstacles.append(new_ob)
                spawn_timer = profile.spawn_interval
            else:
                # Retry on the next frame.
                spawn_timer = min(dt, 0.01)

        # Resolve obstacles and update d exactly once per obstacle.
        obstacles = resolve_obstacles(
            obstacles,
            player_x,
            player_y,
            player_w,
            player_h,
            ground_y,
            speed,
            dt,
            controller,
            config,
        )

        loop_ms = (time.perf_counter() - loop_start) * 1000.0
        loop_ms_tracker.add(loop_ms)

        # Finish logging the previous frame's jump now that loop_ms is known.
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

        draw_text(f"Difficulty preset: {preset_id} (d={controller.d:.2f})", 525, 25)
        pygame.draw.rect(screen, diff_color, (780, 28, 20, 14), border_radius=3)
        draw_text(f"Gesture: {detected} ({score:.2f})", 525, 58)
        draw_text(f"Threshold: {threshold:.2f}  (+ / -)", 525, 91)
        draw_text(f"Speed: {speed:.0f}", 525, 124)
        draw_text(f"FPS: {fps:.1f}  |  Loop: {loop_ms:.1f} ms", 525, 157)
        draw_text(
            "Controls: 1 easy  2 normal  3 hard  D debug  +/- threshold  R restart  Q quit",
            525,
            190,
            use_small=True,
        )

        pygame.draw.rect(screen, (13, 17, 24), (0, 390, WIDTH, HEIGHT - 390))
        pygame.draw.line(screen, (110, 125, 145), (0, ground_y), (WIDTH, ground_y), 3)
        pygame.draw.rect(
            screen,
            (88, 197, 255),
            (int(player_x), int(player_y), player_w, player_h),
            border_radius=7,
        )
        for ob in obstacles:
            color = (120, 120, 120) if ob.failed else (255, 111, 105)
            rect = pygame.Rect(
                int(ob.x), int(ground_y - ob.h), int(ob.w), int(ob.h)
            )
            pygame.draw.rect(screen, color, rect, border_radius=5)

        draw_text(f"Cleared: {controller.successes}  Crashed: {controller.failures}", 24, 410)
        draw_text("Open hand = jump. Close hand to re-arm.", 24, 445, use_small=True)

        if debug_overlay:
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

        pygame.display.flip()

    hands.close()
    cap.release()
    if latency_log is not None:
        latency_log.close()
    pygame.quit()


if __name__ == "__main__":
    main()
