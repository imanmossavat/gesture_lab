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

from adaptive import (
    AdaptiveController,
    Config,
    LatencyTracker,
    Obstacle,
    SessionTracker,
    air_time,
    attempt_jump,
    clears_obstacle,
    difficulty_profile,
    jump_peak,
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
    assert p1.speed == pytest.approx(430.0)
    assert p1.height == pytest.approx(70.0)
    assert p1.width == pytest.approx(35.0)
    assert p1.spawn_interval == pytest.approx(1.3)

    p5 = difficulty_profile(0.5)
    assert p5.speed == pytest.approx(345.0)
    assert p5.height == pytest.approx(50.0)
    assert p5.width == pytest.approx(27.5)
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
    assert spawn_allowed(prev, cfg.SPAWN_X, 430.0, cfg) is True


def test_min_gap_with_frame_time_spike():
    cfg = Config()
    speed = 430.0
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
    # d = 0, worst-case height with +10% jitter
    h = 30.0 * 1.1
    w = 20.0
    speed = 260.0
    assert timing_window(h, w, speed, cfg) > 0.0
    assert clears_obstacle(h, w, speed, 0.0, cfg) is True

    # d = 1, worst-case height with +10% jitter
    h = 70.0 * 1.1
    w = 35.0
    speed = 430.0
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
        h = profile.height * rng.uniform(0.9, 1.1)
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
    cfg = Config()
    rng = random.Random(123)
    profile = difficulty_profile(0.5, cfg)
    for _ in range(20):
        t = next_spawn_timer(profile, rng, cfg)
        assert profile.spawn_interval <= t <= profile.spawn_interval * 1.3


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
    cfg = Config()
    air = air_time(cfg)
    jitter_min = cfg.SPAWN_INTERVAL_JITTER[0]
    expected = [1.656, 1.355, 1.054, 0.753, 0.452]
    for d, exp in zip([0.0, 0.25, 0.5, 0.75, 1.0], expected):
        profile = difficulty_profile(d, cfg)
        gap_px = profile.speed * profile.spawn_interval * jitter_min - profile.width
        slack = gap_px / profile.speed - air
        assert slack == pytest.approx(exp, abs=0.001)


def test_game_outcomes_csv_and_analyze(tmp_path):
    """Smoke-run the game with --log-outcomes and consume the log with analyze.py."""
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
    assert result.returncode == 0, result.stderr
    text = out.read_text()
    header = "t_s,player,session,mode,d_spawn,d_after,speed,height,width,outcome"
    assert header in text, text[:200]
    for key in ("ADAPTATION_RATE", "TARGET_SUCCESS_RATE", "D_START", "effective_start_d",
                "initial_preset", "git_hash", "timestamp"):
        assert f"# {key}=" in text, key
    rows = list(csv.reader(text.splitlines()))
    assert len(rows) > 1

    analyze = subprocess.run(
        ["python", "analyze.py", str(out)],
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
    header = ["t_s", "player", "session", "mode", "d_spawn", "d_after", "speed", "height", "width", "outcome"]
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
        ["python", "analyze.py", str(csv_path)],
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
    from game import GameState, PRESETS
    from adaptive import AdaptiveController, Config, Obstacle

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


def test_analyze_multi_session_plot(tmp_path):
    """analyze.py --plot writes one PNG per session for multi-session files."""
    matplotlib = pytest.importorskip("matplotlib")
    csv_path = tmp_path / "two_sessions.csv"
    header = ["t_s", "player", "session", "mode", "d_spawn", "d_after", "speed", "height", "width", "outcome"]
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
        ["python", "analyze.py", str(csv_path), "--plot", str(plot_base), "--window", "10", "--tail", "10"],
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
            "python", "dev_bench.py", "simulate",
            "--frames", "200", "--noise", "60", "--latency", "50", "--miss", "0.05",
            "--start-d", "0.2", "--adaptation-rate", "0.06", "--target-success-rate", "0.8",
            "--csv", str(csv_path),
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    text = csv_path.read_text()
    assert "t_s,player,session,mode,d_spawn,d_after,speed,height,width,outcome" in text
    for key in ("ADAPTATION_RATE", "TARGET_SUCCESS_RATE", "D_START", "seed", "noise_ms",
                "latency_jitter_ms", "miss_prob", "git_hash", "timestamp"):
        assert f"# {key}=" in text, key
    assert "# ADAPTATION_RATE=0.06" in text
    assert "# TARGET_SUCCESS_RATE=0.8" in text
    assert "# D_START=0.2" in text


def test_analyze_and_summarize_read_metadata(tmp_path):
    """analyze.py and summarize.py surface run metadata from dev_bench CSVs."""
    csv_path = tmp_path / "sim.csv"
    subprocess.run(
        [
            "python", "dev_bench.py", "simulate",
            "--frames", "200", "--adaptation-rate", "0.06", "--target-success-rate", "0.8",
            "--csv", str(csv_path),
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        check=True,
        capture_output=True,
        text=True,
    )

    analyze = subprocess.run(
        ["python", "analyze.py", str(csv_path)],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert analyze.returncode == 0, analyze.stderr
    assert "metadata:" in analyze.stdout
    assert "ADAPTATION_RATE = 0.06" in analyze.stdout
    assert "TARGET_SUCCESS_RATE = 0.8" in analyze.stdout

    summarize = subprocess.run(
        ["python", "summarize.py", str(csv_path)],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert summarize.returncode == 0, summarize.stderr
    assert "ADAPTATION_RATE=0.06" in summarize.stdout
    assert "TARGET_SUCCESS_RATE=0.8" in summarize.stdout


def test_simulate_returns_obstacle_rows():
    """_simulate now returns per-obstacle rows, not just outcome integers."""
    from dev_bench import _simulate
    cfg = Config()
    rng = random.Random(7)
    controller, rows = _simulate(
        cfg, rng, frames=50, timing_noise_ms=40.0, latency_jitter_ms=30.0, miss_prob=0.0, seed=7,
    )
    assert len(rows) == 50
    assert all(r["outcome"] in (0, 1) for r in rows)
    assert all("d_spawn" in r and "d_after" in r for r in rows)
    assert controller.outcomes == 50
