"""Tests for the pure-Python adaptive difficulty system.

These tests require no camera, no display, and no MediaPipe model.
"""
import math
import random

import pytest

from adaptive import (
    AdaptiveController,
    Config,
    LatencyTracker,
    Obstacle,
    air_time,
    attempt_jump,
    clears_obstacle,
    difficulty_profile,
    jump_peak,
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
    assert p1.spawn_interval == pytest.approx(1.5)

    p5 = difficulty_profile(0.5)
    assert p5.speed == pytest.approx(345.0)
    assert p5.height == pytest.approx(50.0)
    assert p5.width == pytest.approx(27.5)
    assert p5.spawn_interval == pytest.approx(2.0)


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
