"""Tests for the pure-Python adaptive difficulty system.

These tests require no camera, no display, and no MediaPipe model.
"""
import csv
import math
import os
import random
import subprocess
from pathlib import Path

import pytest

from gesture_lab.adaptive import (
    AdaptiveController,
    Config,
    LATENCY_HEADER,
    LatencyTracker,
    Obstacle,
    OUTCOME_HEADER,
    PRESETS,
    Phase,
    SessionTracker,
    air_time,
    attempt_jump,
    clears_obstacle,
    difficulty_profile,
    jump_peak,
    next_phase,
    next_spawn_timer,
    resolve_obstacles,
    spawn_allowed,
    time_above,
    timing_window,
    try_spawn,
)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

def test_profile_at_difficulty_boundaries():
    p0 = difficulty_profile(0.0)
    assert p0.speed == pytest.approx(260.0)
    assert p0.height == pytest.approx(30.0)
    assert p0.width == pytest.approx(20.0)
    assert p0.spawn_interval == pytest.approx(2.5)

    p1 = difficulty_profile(1.0)
    assert p1.speed == pytest.approx(550.0)
    assert p1.height == pytest.approx(90.0)
    assert p1.width == pytest.approx(50.0)
    assert p1.spawn_interval == pytest.approx(1.3)

    p5 = difficulty_profile(0.5)
    assert p5.speed == pytest.approx(405.0)
    assert p5.height == pytest.approx(60.0)
    assert p5.width == pytest.approx(35.0)
    assert p5.spawn_interval == pytest.approx(1.9)


def test_profile_clamps_out_of_range_d():
    low = difficulty_profile(-0.5)
    high = difficulty_profile(2.0)
    assert low == difficulty_profile(0.0)
    assert high == difficulty_profile(1.0)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

def test_success_and_failure_deltas():
    c = AdaptiveController()
    c.update(1)
    assert c.d == pytest.approx(0.52)  # +0.02
    c.update(0)
    assert c.d == pytest.approx(0.46)  # -0.06


def test_zero_momentum_matches_original_update():
    c = AdaptiveController(Config(MOMENTUM=0.0))
    for outcome, expected in [(1, 0.52), (0, 0.46), (1, 0.48)]:
        c.update(outcome)
        assert c.d == pytest.approx(expected)


def test_momentum_first_step_matches_plain_update_then_keeps_moving():
    plain = AdaptiveController(Config(MOMENTUM=0.0))
    mom = AdaptiveController(Config(MOMENTUM=0.8))
    plain.update(0)
    mom.update(0)
    assert mom.d == pytest.approx(plain.d)  # velocity starts at zero
    plain.update(0)
    mom.update(0)
    assert mom.d < plain.d  # accumulated velocity pushes further


def test_momentum_carries_through_after_streak_ends():
    c = AdaptiveController(Config(MOMENTUM=0.8))
    for _ in range(3):  # keep d away from the lower bound
        c.update(0)
    d_before = c.d
    c.update(1)  # a success, but velocity is still negative
    assert c.d < d_before


def test_momentum_long_run_gain_is_eta_over_one_minus_beta():
    # Learning rate and momentum are independent; a streak settles at eta/(1-beta).
    c = AdaptiveController(Config(MOMENTUM=0.8, D_START=0.0))
    c = AdaptiveController(Config(MOMENTUM=0.8, D_START=0.0, D_MAX=100.0))
    for _ in range(60):
        c.update(1)
    assert c.velocity == pytest.approx(0.08 * 0.25 / (1 - 0.8), rel=1e-2)


def test_momentum_velocity_zeroed_at_bounds_and_reset():
    c = AdaptiveController(Config(MOMENTUM=0.9, D_START=0.0))
    c.update(0)
    assert c.d == 0.0 and c.velocity == 0.0
    c.update(1)
    c.velocity = 0.3
    c.reset()
    assert c.velocity == 0.0


def test_controller_clamps_at_bounds():
    c = AdaptiveController(Config(D_START=1.0))
    c.update(1)
    assert c.d == pytest.approx(1.0)
    c = AdaptiveController(Config(D_START=0.0))
    c.update(0)
    assert c.d == pytest.approx(0.0)


def test_one_success_counted_once():
    c = AdaptiveController()
    ob = Obstacle(x=500.0, w=30.0, h=50.0)
    cfg = Config()
    ob.x = 700.0
    resolve_obstacles(
        [ob], 620.0, 534.0, 42.0, 56.0,
        cfg.GROUND_Y, speed=300.0, dt=0.5, controller=c, config=cfg,
    )
    assert c.successes == 1
    assert c.outcomes == 1
    resolve_obstacles(
        [ob], 620.0, 534.0, 42.0, 56.0,
        cfg.GROUND_Y, speed=300.0, dt=0.5, controller=c, config=cfg,
    )
    assert c.successes == 1
    assert c.outcomes == 1


def test_one_collision_counted_once():
    c = AdaptiveController()
    ob = Obstacle(x=620.0, w=30.0, h=50.0)
    cfg = Config()
    resolve_obstacles(
        [ob], 620.0, 534.0, 42.0, 56.0,
        cfg.GROUND_Y, speed=300.0, dt=0.016, controller=c, config=cfg,
    )
    assert c.failures == 1
    assert c.outcomes == 1
    assert ob.done
    assert ob.failed
    resolve_obstacles(
        [ob], 620.0, 534.0, 42.0, 56.0,
        cfg.GROUND_Y, speed=300.0, dt=0.016, controller=c, config=cfg,
    )
    assert c.failures == 1
    assert c.outcomes == 1


def test_collision_followed_by_next_obstacle():
    c = AdaptiveController()
    cfg = Config()
    # First obstacle collides while the player is on the ground.
    ob1 = Obstacle(x=620.0, w=30.0, h=50.0)
    resolve_obstacles(
        [ob1], 620.0, 534.0, 42.0, 56.0,
        cfg.GROUND_Y, speed=300.0, dt=0.016, controller=c, config=cfg,
    )
    assert c.failures == 1

    # Second obstacle passes while the player is in the air (jumping).
    ob2 = Obstacle(x=1100.0, w=30.0, h=40.0)
    resolve_obstacles(
        [ob2], 620.0, 434.0, 42.0, 56.0,
        cfg.GROUND_Y, speed=300.0, dt=2.0, controller=c, config=cfg,
    )
    assert c.successes == 1
    assert c.outcomes == 2


# ---------------------------------------------------------------------------
# Spawn / min-gap
# ---------------------------------------------------------------------------

def test_min_gap_respected_at_limit():
    cfg = Config()
    speed = 300.0
    required = max(
        cfg.MIN_GAP_FLOOR_PX,
        speed * (air_time(cfg) + cfg.LANDING_MARGIN_S),
    )
    prev = Obstacle(x=cfg.SPAWN_X - 100.0 - 30.0, w=30.0, h=50.0)
    # Exactly the required gap: allowed.
    prev.x = cfg.SPAWN_X - required - prev.w
    assert spawn_allowed(prev, cfg.SPAWN_X, speed, cfg) is True
    # Just below: not allowed.
    prev.x = cfg.SPAWN_X - required - prev.w + 1.0
    assert spawn_allowed(prev, cfg.SPAWN_X, speed, cfg) is False


def test_min_gap_ignores_off_screen_previous_obstacle():
    cfg = Config()
    prev = Obstacle(x=-200.0, w=30.0, h=50.0)
    assert spawn_allowed(prev, cfg.SPAWN_X, cfg.SPEED_MAX, cfg) is True


def test_min_gap_with_frame_time_spike():
    cfg = Config()
    speed = cfg.SPEED_MAX
    # Simulate a large dt that would shrink the gap dangerously.
    prev = Obstacle(x=cfg.SPAWN_X - 50.0, w=30.0, h=50.0)
    assert spawn_allowed(prev, cfg.SPAWN_X, speed, cfg) is False


def test_spawn_returns_none_when_gap_too_small():
    cfg = Config()
    rng = random.Random(1)
    profile = difficulty_profile(1.0, cfg)
    prev = Obstacle(x=cfg.SPAWN_X - 50.0, w=30.0, h=50.0)
    assert try_spawn(prev, profile, profile.speed, rng, cfg) is None


def test_spawn_returns_obstacle_when_allowed():
    cfg = Config()
    rng = random.Random(1)
    profile = difficulty_profile(0.0, cfg)
    ob = try_spawn(None, profile, profile.speed, rng, cfg)
    assert ob is not None
    assert ob.x == pytest.approx(cfg.SPAWN_X)
    assert ob.w == pytest.approx(profile.width)
    assert ob.h != pytest.approx(profile.height)  # jittered


def test_multiple_obstacles_at_once_resolved():
    c = AdaptiveController()
    cfg = Config()
    obs = [
        Obstacle(x=850.0, w=30.0, h=40.0),
        Obstacle(x=950.0, w=30.0, h=40.0),
    ]
    resolve_obstacles(
        obs, 620.0, 534.0, 42.0, 56.0,
        cfg.GROUND_Y, speed=300.0, dt=2.0, controller=c, config=cfg,
    )
    assert c.outcomes == 2
    assert all(ob.done for ob in obs)


# ---------------------------------------------------------------------------
# Jump
# ---------------------------------------------------------------------------

def test_jump_accepted_on_ground():
    cfg = Config()
    vy, accepted = attempt_jump(
        player_y=cfg.GROUND_Y - cfg.PLAYER_HEIGHT,
        velocity_y=0.0,
        ground_y=cfg.GROUND_Y,
        player_h=cfg.PLAYER_HEIGHT,
        jump_velocity=cfg.JUMP_VELOCITY,
    )
    assert accepted is True
    assert vy == pytest.approx(cfg.JUMP_VELOCITY)


def test_jump_rejected_when_airborne():
    cfg = Config()
    vy, accepted = attempt_jump(
        player_y=cfg.GROUND_Y - cfg.PLAYER_HEIGHT - 50.0,
        velocity_y=-400.0,
        ground_y=cfg.GROUND_Y,
        player_h=cfg.PLAYER_HEIGHT,
        jump_velocity=cfg.JUMP_VELOCITY,
    )
    assert accepted is False
    assert vy == pytest.approx(-400.0)


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------

def test_latency_tracker_empty():
    t = LatencyTracker(10)
    assert t.count == 0
    assert t.mean() == 0.0
    assert t.median() == 0.0
    assert t.p90() == 0.0
    assert t.std() == 0.0


def test_latency_tracker_one_sample():
    t = LatencyTracker(10)
    t.add(42.0)
    assert t.mean() == pytest.approx(42.0)
    assert t.median() == pytest.approx(42.0)
    assert t.p90() == pytest.approx(42.0)
    assert t.std() == 0.0


def test_latency_tracker_two_samples():
    t = LatencyTracker(10)
    t.add(10.0)
    t.add(20.0)
    assert t.mean() == pytest.approx(15.0)
    assert t.median() == pytest.approx(15.0)
    assert t.p90() == pytest.approx(19.0)
    assert t.std() == pytest.approx(7.071, rel=1e-3)


# ---------------------------------------------------------------------------
# Physics / feasibility
# ---------------------------------------------------------------------------

def test_theoretical_physics_limits():
    cfg = Config()
    assert jump_peak(cfg) == pytest.approx(132.25)
    assert air_time(cfg) == pytest.approx(0.766667, rel=1e-5)
    assert time_above(0.0, cfg) == pytest.approx(air_time(cfg))
    assert time_above(jump_peak(cfg), cfg) == pytest.approx(0.0)


def test_feasibility_at_extremes_with_height_jitter():
    cfg = Config()
    # d = 0, design-margin height with +2σ jitter
    h = 30.0 * (cfg.HEIGHT_JITTER_MEAN + 2.0 * cfg.HEIGHT_JITTER_STD)
    w = 20.0
    speed = 260.0
    assert timing_window(h, w, speed, cfg) > 0.0
    assert clears_obstacle(h, w, speed, 0.0, cfg) is True

    # d = 1, design-margin height with +2σ jitter (use Config ceiling values)
    h = cfg.HEIGHT_MAX * (cfg.HEIGHT_JITTER_MEAN + 2.0 * cfg.HEIGHT_JITTER_STD)
    w = cfg.WIDTH_MAX
    speed = cfg.SPEED_MAX
    assert timing_window(h, w, speed, cfg) > 0.0
    assert clears_obstacle(h, w, speed, 0.0, cfg) is True


# ---------------------------------------------------------------------------
# Seeded simulation
# ---------------------------------------------------------------------------

def _run_simulation(seed, n, timing_noise_ms, latency_jitter_ms, miss_prob, skill_bias_ms=0.0):
    cfg = Config()
    rng = random.Random(seed)
    c = AdaptiveController(cfg, rng=random.Random(seed))
    outcomes = []
    for _ in range(n):
        profile = difficulty_profile(c.d, cfg)
        h = profile.height * max(0.1, rng.gauss(1.0, cfg.HEIGHT_JITTER_STD))
        w = profile.width
        speed = profile.speed
        if rng.random() < miss_prob:
            success = False
        else:
            sigma = math.hypot(timing_noise_ms, latency_jitter_ms)
            offset_ms = rng.gauss(skill_bias_ms, sigma)
            success = clears_obstacle(h, w, speed, offset_ms / 1000.0, cfg)
        c.update(1 if success else 0)
        outcomes.append(1 if success else 0)
    return c, outcomes


def test_seeded_simulation_steady_state_success_near_target():
    """A mid-skill simulated player should converge near P_TARGET = 0.75."""
    c, outcomes = _run_simulation(
        seed=42,
        n=2000,
        timing_noise_ms=120.0,
        latency_jitter_ms=90.0,
        miss_prob=0.05,
    )
    overall_rate = sum(outcomes) / len(outcomes)
    last_rate = sum(outcomes[-400:]) / 400
    # Tolerance chosen to allow normal simulation variance; if this fails,
    # it indicates a bug, not a calibration decision.
    assert 0.70 <= overall_rate <= 0.80
    assert 0.72 <= last_rate <= 0.79
    assert 0.3 <= c.d <= 0.8


def test_done_obstacle_keeps_moving_after_outcome():
    """Regression: obstacles must scroll off-screen after being marked done."""
    c = AdaptiveController()
    ob = Obstacle(x=700.0, w=30.0, h=50.0)
    cfg = Config()
    resolve_obstacles(
        [ob], 620.0, 534.0, 42.0, 56.0,
        cfg.GROUND_Y, speed=300.0, dt=0.5, controller=c, config=cfg,
    )
    assert ob.done
    assert c.outcomes == 1
    x_after_success = ob.x
    resolve_obstacles(
        [ob], 620.0, 534.0, 42.0, 56.0,
        cfg.GROUND_Y, speed=300.0, dt=0.1, controller=c, config=cfg,
    )
    assert c.outcomes == 1  # no double counting
    assert ob.x < x_after_success


def test_next_spawn_timer_within_jitter_range():
    cfg = Config(SPAWN_GAP_MODE="jitter")
    rng = random.Random(123)
    profile = difficulty_profile(0.5, cfg)
    for _ in range(20):
        t = next_spawn_timer(profile, rng, cfg)
        mean = cfg.SPAWN_INTERVAL_JITTER_MEAN
        std = cfg.SPAWN_INTERVAL_JITTER_STD
        lo = profile.spawn_interval * (mean - 3.0 * std)
        hi = profile.spawn_interval * (mean + 3.0 * std)
        assert lo <= t <= hi


def test_wild_spawn_timer_is_feasible_bounded_and_keeps_mean():
    """Wild mode: never below the feasible minimum, bounded above, mean preserved."""
    from gesture_lab.adaptive import min_spawn_interval
    cfg = Config(SPAWN_GAP_MODE="wild")
    rng = random.Random(7)
    for d in (0.0, 0.5, 0.8, 1.0):
        profile = difficulty_profile(d, cfg)
        lo = min_spawn_interval(profile, cfg)
        mean = max(profile.spawn_interval, lo + cfg.SPAWN_WILD_ROOM_S)
        xs = [next_spawn_timer(profile, rng, cfg) for _ in range(4000)]
        assert min(xs) >= lo - 1e-9
        assert max(xs) <= 2.0 * mean - lo + 1e-9
        assert sum(xs) / len(xs) == pytest.approx(mean, rel=0.03)
        # unpredictable: spread far larger than a player's ~0.1 s timing noise
        sd = (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5
        assert sd > 0.25


def test_wild_spawn_pairs_pass_the_game_gap_rule():
    """Gaps from the wild sampler satisfy spawn_allowed at the moment the timer fires."""
    cfg = Config(SPAWN_GAP_MODE="wild")
    rng = random.Random(11)
    for d in (0.0, 0.5, 1.0):
        profile = difficulty_profile(d, cfg)
        for _ in range(500):
            t = next_spawn_timer(profile, rng, cfg)
            prev = Obstacle(x=cfg.SPAWN_X - profile.speed * t, w=profile.width,
                            h=profile.height, done=False, failed=False)
            assert spawn_allowed(prev, cfg.SPAWN_X, profile.speed, cfg)


def test_spawn_mode_rejects_unknown_value():
    cfg = Config(SPAWN_GAP_MODE="nope")
    with pytest.raises(ValueError):
        next_spawn_timer(difficulty_profile(0.5, cfg), random.Random(1), cfg)


def test_on_outcome_callback_receives_single_event():
    c = AdaptiveController()
    recorded = []

    def cb(ob, outcome):
        recorded.append((ob, outcome))

    ob = Obstacle(x=700.0, w=30.0, h=50.0)
    cfg = Config()
    resolve_obstacles(
        [ob], 620.0, 534.0, 42.0, 56.0,
        cfg.GROUND_Y, speed=300.0, dt=0.5, controller=c, config=cfg,
        on_outcome=cb,
    )
    assert len(recorded) == 1
    assert recorded[0][0] is ob
    assert recorded[0][1] == 1


def test_feasibility_schedule_slack_values():
    """Schedule-mode slacks use the design-margin jitter floor and match pinned values."""
    cfg = Config()
    jitter_floor = cfg.SPAWN_INTERVAL_JITTER_MEAN - 2.0 * cfg.SPAWN_INTERVAL_JITTER_STD
    expected = {
        0.00: 1.3064,
        0.25: 1.0426,
        0.50: 0.7809,
        0.75: 0.5203,
        1.00: 0.2604,
    }
    for d, expected_slack in expected.items():
        profile = difficulty_profile(d, cfg)
        gap_px = profile.speed * profile.spawn_interval * jitter_floor - profile.width
        slack = gap_px / profile.speed - air_time(cfg)
        assert slack == pytest.approx(expected_slack, abs=0.001)


def test_set_preset_resets_controller_failures():
    """set_preset() resets the controller counters before applying the new preset."""
    pytest.importorskip("pygame")
    pytest.importorskip("cv2")
    from gesture_lab.game import GameState

    controller = AdaptiveController(Config())
    controller.update(0)
    controller.update(0)
    controller.update(0)
    assert controller.failures == 3

    state = GameState(controller, Config(), start_preset=1)
    state.set_preset(2)
    assert controller.failures == 0
    assert state.preset_id == 2


def _require_game_ran(result):
    """The headless game run needs the hand model (one download). Skip, don't fail, without it."""
    if result.returncode != 0 and "hand_landmarker.task" in result.stderr:
        pytest.skip("hand model not available (needs one download)")
    assert result.returncode == 0, result.stderr


def test_game_outcomes_csv_and_analyze(tmp_path):
    """Smoke-run the game with --log-outcomes and consume the log with analyze.py."""
    pytest.importorskip("pygame")
    pytest.importorskip("cv2")
    out = tmp_path / "o.csv"
    env = os.environ.copy()
    env["SDL_VIDEODRIVER"] = "dummy"
    result = subprocess.run(
        ["python", "game.py", "--dummy-camera", "--smoke-frames", "600",
         "--log-outcomes", str(out), "--player", "test"],
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
        capture_output=True,
        text=True,
    )
    _require_game_ran(result)
    text = out.read_text()
    header = ",".join(OUTCOME_HEADER)
    assert header in text, text[:200]
    for key in ("ADAPTATION_RATE", "TARGET_SUCCESS_RATE", "D_START", "effective_start_d",
                "initial_preset", "git_hash", "timestamp"):
        assert f"# {key}=" in text, key
    rows = list(csv.reader(text.splitlines()))
    assert len(rows) > 1

    analyze = subprocess.run(
        ["python", "-m", "gesture_lab.analyze", str(out)],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert analyze.returncode == 0, analyze.stderr
    assert "n=" in analyze.stdout or "outcome" in analyze.stdout.lower()

# ---------------------------------------------------------------------------
# Sessions / analyze integration
# ---------------------------------------------------------------------------

def test_session_tracker_increments_monotonically():
    tracker = SessionTracker()
    assert tracker.current == 0
    assert tracker.bump() == 1
    assert tracker.bump() == 2
    assert tracker.current == 2


def test_analyze_per_session_and_combined_summary(tmp_path):
    """analyze.py splits by session and reports tail/settling per session."""
    csv_path = tmp_path / "two_sessions.csv"
    header = OUTCOME_HEADER
    rows = []
    # Session 1: d stays near 0.5, all successes -> should settle from obstacle #0.
    for i in range(20):
        d = 0.5 + i * 0.002
        rows.append([i, "test", 1, 2, f"{d:.4f}", f"{d:.4f}", 300, 50, 30, 1])
    # Session 2: d starts near 0.9 (big jump from session 1) and stays there.
    for i in range(15):
        d = 0.9 - i * 0.002
        rows.append([20 + i, "test", 2, 3, f"{d:.4f}", f"{d:.4f}", 400, 65, 33, 1])
    lines = [",".join(str(c) for c in header)]
    lines += [",".join(str(c) for c in r) for r in rows]
    csv_path.write_text("\n".join(lines) + "\n")

    result = subprocess.run(
        ["python", "-m", "gesture_lab.analyze", str(csv_path)],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout

    assert "session 1" in out
    assert "session 2" in out
    assert "all sessions" in out
    # Wilson upper bound should be clamped at 1.000 because every row is a success.
    assert "1.000]" in out

    # Split the report into per-block lines to inspect per-session settling.
    blocks = {}
    current = None
    for line in out.splitlines():
        if line.startswith("=== "):
            current = line
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)

    def block_with(name):
        for key, val in blocks.items():
            if name in key:
                return "\n".join(val)
        return ""

    sess1 = block_with("session 1")
    sess2 = block_with("session 2")
    assert "from obstacle #0" in sess1, sess1
    assert "from obstacle #0" in sess2, sess2


# ---------------------------------------------------------------------------
# Game state / preset boundary
# ---------------------------------------------------------------------------

def test_preset_change_clears_obstacles():
    """A preset change must start a new session with no leftover obstacles."""
    pytest.importorskip("pygame")
    pytest.importorskip("cv2")
    from gesture_lab.game import GameState
    from gesture_lab.adaptive import PRESETS
    from gesture_lab.adaptive import AdaptiveController, Config, Obstacle

    config = Config()
    controller = AdaptiveController(config)
    state = GameState(controller, config)

    # Seed some obstacles as if they were spawned mid-session.
    state.obstacles = [
        Obstacle(100.0, 50.0, 30.0, 300.0, 0.5, False),
        Obstacle(200.0, 55.0, 35.0, 320.0, 0.6, False),
    ]
    state.spawn_timer = 0.123
    previous_session = state.tracker.current

    state.set_preset(3)

    assert state.tracker.current == previous_session + 1
    assert state.obstacles == []
    assert state.spawn_timer == pytest.approx(0.8)
    assert controller.d == PRESETS[3]["d"]
    assert state.threshold == PRESETS[3]["threshold"]
    assert state.stability == PRESETS[3]["stability"]


def test_game_state_uses_config_geometry():
    """GameState geometry/physics must come from the passed Config."""
    pytest.importorskip("pygame")
    pytest.importorskip("cv2")
    from gesture_lab.game import GameState
    from gesture_lab.adaptive import AdaptiveController, Config

    cfg = Config(
        PLAYER_X=111.0, PLAYER_WIDTH=22.0, PLAYER_HEIGHT=33.0,
        GROUND_Y=444.0, GRAVITY=1234.0, JUMP_VELOCITY=-555.0,
    )
    controller = AdaptiveController(cfg)
    state = GameState(controller, cfg)
    assert state.player_x == cfg.PLAYER_X
    assert state.player_w == cfg.PLAYER_WIDTH
    assert state.player_h == cfg.PLAYER_HEIGHT
    assert state.ground_y == cfg.GROUND_Y
    assert state.gravity == cfg.GRAVITY
    assert state.jump_velocity == cfg.JUMP_VELOCITY
    assert state.player_y == cfg.GROUND_Y - cfg.PLAYER_HEIGHT


def test_analyze_multi_session_plot(tmp_path):
    """analyze.py --plot writes one PNG per session for multi-session files."""
    matplotlib = pytest.importorskip("matplotlib")
    csv_path = tmp_path / "two_sessions.csv"
    header = OUTCOME_HEADER
    rows = []
    # Session 1: varying d across bins, mixed outcomes.
    for i in range(25):
        d = 0.25 + 0.5 * (i / 24.0)
        rows.append([i, "test", 1, 2, f"{d:.4f}", f"{d:.4f}", 300, 50, 30, 1 if i % 3 else 0])
    # Session 2: different d range, mixed outcomes.
    for i in range(25):
        d = 0.6 + 0.3 * (i / 24.0)
        rows.append([25 + i, "test", 2, 3, f"{d:.4f}", f"{d:.4f}", 350, 60, 32, 1 if i % 4 else 0])
    lines = [",".join(str(c) for c in header)]
    lines += [",".join(str(c) for c in r) for r in rows]
    csv_path.write_text("\n".join(lines) + "\n")

    plot_base = tmp_path / "report.png"
    result = subprocess.run(
        ["python", "-m", "gesture_lab.analyze", str(csv_path), "--plot", str(plot_base), "--window", "10", "--tail", "10"],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "report_s1.png").exists()
    assert (tmp_path / "report_s2.png").exists()
    # Pooled summary should not produce a plot file.
    assert not (tmp_path / "report_all sessions.png").exists()
    assert not (tmp_path / "report_all_sessions.png").exists()



# ---------------------------------------------------------------------------
# Run metadata in CSV outputs
# ---------------------------------------------------------------------------

def test_dev_bench_simulate_writes_metadata(tmp_path):
    """dev_bench simulate --csv emits config metadata as leading comments."""
    csv_path = tmp_path / "sim.csv"
    result = subprocess.run(
        [
            "python", "-m", "gesture_lab.dev_bench", "simulate",
            "--frames", "200", "--noise", "60", "--latency", "50", "--miss", "0.05",
            "--start-d", "0.2", "--adaptation-rate", "0.06", "--target-success-rate", "0.8",
            "--momentum", "0.5",
            "--csv", str(csv_path),
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    text = csv_path.read_text()
    assert ",".join(OUTCOME_HEADER) in text
    for key in ("ADAPTATION_RATE", "TARGET_SUCCESS_RATE", "D_START", "seed", "noise_ms",
                "latency_jitter_ms", "miss_prob", "git_hash", "timestamp"):
        assert f"# {key}=" in text, key
    assert "# ADAPTATION_RATE=0.06" in text
    assert "# TARGET_SUCCESS_RATE=0.8" in text
    assert "# D_START=0.2" in text
    assert "# MOMENTUM=0.5" in text


def test_analyze_and_summarize_read_metadata(tmp_path):
    """analyze.py and summarize.py surface run metadata from dev_bench CSVs."""
    csv_path = tmp_path / "sim.csv"
    subprocess.run(
        [
            "python", "-m", "gesture_lab.dev_bench", "simulate",
            "--frames", "200", "--adaptation-rate", "0.06", "--target-success-rate", "0.8",
            "--momentum", "0.5",
            "--csv", str(csv_path),
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        check=True,
        capture_output=True,
        text=True,
    )

    analyze = subprocess.run(
        ["python", "-m", "gesture_lab.analyze", str(csv_path)],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert analyze.returncode == 0, analyze.stderr
    assert "metadata:" in analyze.stdout
    assert "ADAPTATION_RATE = 0.06" in analyze.stdout
    assert "MOMENTUM = 0.5" in analyze.stdout
    assert "TARGET_SUCCESS_RATE = 0.8" in analyze.stdout

    summarize = subprocess.run(
        ["python", "-m", "gesture_lab.summarize", str(csv_path)],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert summarize.returncode == 0, summarize.stderr
    assert "ADAPTATION_RATE=0.06" in summarize.stdout
    assert "MOMENTUM=0.5" in summarize.stdout
    assert "TARGET_SUCCESS_RATE=0.8" in summarize.stdout


def test_simulate_returns_obstacle_rows():
    """_simulate now returns per-obstacle rows, not just outcome integers."""
    from gesture_lab.dev_bench import _simulate
    cfg = Config()
    rng = random.Random(7)
    controller, rows = _simulate(
        cfg, rng, frames=50, timing_noise_ms=40.0, latency_jitter_ms=30.0, miss_prob=0.0, seed=7,
    )
    assert len(rows) == 50
    assert all(r["outcome"] in (0, 1) for r in rows)
    assert all("d_spawn" in r and "d_after" in r for r in rows)
    assert controller.outcomes == 50


# ---------------------------------------------------------------------------
# Spec-pinning tests
# ---------------------------------------------------------------------------

def test_readme_presets_match_code():
    """README difficulty-preset table must match adaptive.PRESETS."""
    import re

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    table_match = re.search(
        r"## Difficulty presets\s*\n\|[^\n]*\|\n\|[-:| ]+\|\n(.*?)(?=\n##|\Z)",
        readme,
        re.DOTALL,
    )
    assert table_match, "Could not find preset table in README"
    name_to_id = {"easy": 1, "normal": 2, "hard": 3}
    parsed = {}
    for line in table_match.group(1).strip().splitlines():
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) != 4:
            continue
        name, d, stability, threshold = parts
        pid = name_to_id[name.lower()]
        parsed[pid] = {
            "d": float(d),
            "stability": int(stability),
            "threshold": float(threshold),
        }
    for pid in (1, 2, 3):
        assert parsed[pid]["d"] == PRESETS[pid]["d"]
        assert parsed[pid]["stability"] == PRESETS[pid]["stability"]
        assert parsed[pid]["threshold"] == PRESETS[pid]["threshold"]


def test_dev_bench_simulate_header_matches_constant(tmp_path):
    """dev_bench simulate --csv writes the exact OUTCOME_HEADER columns."""
    csv_path = tmp_path / "rep.csv"
    result = subprocess.run(
        ["python", "-m", "gesture_lab.dev_bench", "simulate", "--frames", "10", "--csv", str(csv_path)],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = csv_path.read_text().splitlines()
    header_line = next(line for line in lines if not line.startswith("#"))
    assert header_line == ",".join(OUTCOME_HEADER)


def test_game_latency_header_matches_constant(tmp_path):
    """game.py --log-latency writes the exact LATENCY_HEADER columns."""
    pytest.importorskip("pygame")
    pytest.importorskip("cv2")
    out = tmp_path / "lat.csv"
    env = os.environ.copy()
    env["SDL_VIDEODRIVER"] = "dummy"
    result = subprocess.run(
        ["python", "game.py", "--dummy-camera", "--smoke-frames", "60",
         "--log-latency", str(out), "--player", "test"],
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
        capture_output=True,
        text=True,
    )
    _require_game_ran(result)
    lines = out.read_text().splitlines()
    header_line = next(line for line in lines if not line.startswith("#"))
    assert header_line == ",".join(LATENCY_HEADER)

# ---------------------------------------------------------------------------
# Phase flow
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phase,event,expected",
    [
        (Phase.READY, "space", Phase.PLAYING),
        (Phase.READY, "game_over", Phase.READY),
        (Phase.READY, "preset", Phase.READY),
        (Phase.PLAYING, "space", Phase.PAUSED),
        (Phase.PLAYING, "game_over", Phase.OVER),
        (Phase.PLAYING, "preset", Phase.READY),
        (Phase.PAUSED, "space", Phase.PLAYING),
        (Phase.PAUSED, "game_over", Phase.PAUSED),
        (Phase.PAUSED, "preset", Phase.READY),
        (Phase.OVER, "space", Phase.PLAYING),
        (Phase.OVER, "game_over", Phase.OVER),
        (Phase.OVER, "preset", Phase.READY),
    ],
)
def test_next_phase_table(phase, event, expected):
    assert next_phase(phase, event) is expected



# ---------------------------------------------------------------------------
# Probe mode (tight gaps for measuring recovery time) and gap_s logging
# ---------------------------------------------------------------------------

def test_probe_spawn_timer_is_air_time_plus_rest_range():
    cfg = Config(SPAWN_GAP_MODE="probe", SPAWN_REST_LO_S=0.05, SPAWN_REST_HI_S=0.45)
    rng = random.Random(3)
    profile = difficulty_profile(0.5, cfg)
    xs = [next_spawn_timer(profile, rng, cfg) for _ in range(3000)]
    air = air_time(cfg)
    assert min(xs) >= air + 0.05 - 1e-9
    assert max(xs) <= air + 0.45 + 1e-9
    assert sum(xs) / len(xs) == pytest.approx(air + 0.25, abs=0.02)


def test_probe_mode_allows_gaps_below_the_normal_floor_but_not_overlap():
    prof = difficulty_profile(0.5, Config())
    prev = Obstacle(x=1190.0 - 150.0 - prof.width, w=prof.width, h=50.0)  # 150 px gap
    assert not spawn_allowed(prev, 1190.0, prof.speed, Config(SPAWN_GAP_MODE="wild"))
    assert spawn_allowed(prev, 1190.0, prof.speed, Config(SPAWN_GAP_MODE="probe"))
    overlapping = Obstacle(x=1190.0 - 10.0, w=prof.width, h=50.0)
    assert not spawn_allowed(overlapping, 1190.0, prof.speed, Config(SPAWN_GAP_MODE="probe"))


def test_outcome_header_ends_with_gap_s_and_obstacle_defaults():
    assert OUTCOME_HEADER[-1] == "gap_s"
    assert Obstacle(x=0.0, w=1.0, h=1.0).gap_s == 0.0


def _write_rest_csv(path, rng, penalty_below_s, n=300):
    cfg = Config()
    air = air_time(cfg)
    p = difficulty_profile(0.5, cfg)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTCOME_HEADER)
        w.writeheader()
        for i in range(n):
            rest = rng.uniform(0.0, 0.9)
            h = p.height
            budget = 0.5 * timing_window(h, p.width, p.speed, cfg)
            sigma = 0.12 * (1.0 + (2.5 * max(0.0, (penalty_below_s - rest) / penalty_below_s)
                                   if penalty_below_s else 0.0))
            outcome = 1 if abs(rng.gauss(0.0, sigma)) <= budget else 0
            w.writerow({
                "t_s": f"{i * 1.5:.3f}", "player": "sim", "session": 1, "mode": 2,
                "d_spawn": "0.5000", "d_after": "0.5000", "speed": f"{p.speed:.1f}",
                "height": f"{h:.1f}", "width": f"{p.width:.1f}", "outcome": outcome,
                "gap_s": f"{air + rest:.3f}",
            })


def test_summarize_rest_table_flags_a_recovery_penalty(tmp_path):
    rng = random.Random(21)
    with_penalty = tmp_path / "penalty.csv"
    no_penalty = tmp_path / "plain.csv"
    _write_rest_csv(with_penalty, rng, penalty_below_s=0.3)
    _write_rest_csv(no_penalty, rng, penalty_below_s=0.0)
    root = Path(__file__).resolve().parent.parent

    r1 = subprocess.run(["python", "-m", "gesture_lab.summarize", str(with_penalty)], cwd=str(root),
                        capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    assert "success by rest time" in r1.stdout
    assert "<< below model" in r1.stdout

    r2 = subprocess.run(["python", "-m", "gesture_lab.summarize", str(no_penalty)], cwd=str(root),
                        capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    assert "success by rest time" in r2.stdout
    assert "<< below model" not in r2.stdout


def test_summarize_skips_rest_table_without_gap_column(tmp_path):
    p = tmp_path / "old.csv"
    _write_rest_csv(p, random.Random(5), penalty_below_s=0.0, n=60)
    rows = list(csv.DictReader(open(p)))
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[c for c in OUTCOME_HEADER if c != "gap_s"])
        w.writeheader()
        for r in rows:
            r.pop("gap_s")
            w.writerow(r)
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run(["python", "-m", "gesture_lab.summarize", str(p)], cwd=str(root),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "success by rest time" not in r.stdout


def test_fixed_steps_keeps_game_time_equal_to_real_time():
    """Whatever the camera frame time, physics runs in whole 1/60 s steps and
    game time never drifts from real time by more than one step."""
    from gesture_lab.adaptive import fixed_steps

    step = 1.0 / 60.0
    for frame_s in (1 / 30, 1 / 60, 1 / 45, 0.025, 0.05):
        accum, steps = 0.0, 0
        for i in range(1, 1201):
            n, accum = fixed_steps(accum, frame_s, step)
            steps += n
            assert abs(steps * step - i * frame_s) <= step + 1e-9
        assert 0.0 <= accum < step + 1e-9
