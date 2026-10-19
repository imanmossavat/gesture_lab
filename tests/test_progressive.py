"""Tests for the progressive plan (speed ladder + staircase).  Pure logic."""
import random

from adaptive import Config
from experiment import game_window
from progressive import W_MAX, W_MIN, ProgressivePlan


def _plan(seed=1, **kw):
    return ProgressivePlan(Config(), random.Random(seed), **kw)


def _drain(plan, cleared_fn=lambda t: True):
    trials = []
    while (t := plan.take_next()) is not None:
        trials.append(t)
        plan.report(t.index, cleared_fn(t))
    return trials


def test_plan_has_expected_length_and_ends():
    plan = _plan()
    trials = _drain(plan)
    assert len(trials) == plan.total == 3 + 3 * 10 and plan.exhausted
    assert trials[-1].interval_after is None and trials[-1].gap_after_px is None
    assert all(t.interval_after is not None for t in trials[:-1])


def test_every_obstacle_is_feasible_and_within_window_limits():
    cfg = Config()
    for seed in range(20):
        for fn in (lambda t: True, lambda t: False, lambda t: t.index % 3 == 0):
            for t in _drain(_plan(seed), fn):
                assert W_MIN - 1e-9 <= t.target_w <= W_MAX + 1e-9
                w = game_window(t.height, t.width, t.speed, t.gap_after_px, cfg)
                assert w.feasible and w.width > 0.04          # a real, if narrow, safe window


def test_speed_rises_by_level_and_warmup_is_slow_and_easy():
    trials = _drain(_plan(speed0=400, speed_step=60, levels=3, per_level=10))
    assert [t.speed for t in trials[:13]] == [400.0] * 13          # warm-up + first level
    assert trials[13].speed == 460.0 and trials[-1].speed == 520.0
    assert all(t.target_w == W_MAX for t in trials[:3])


def test_staircase_shrinks_after_clears_and_grows_after_crashes():
    plan = _plan(seed=3)
    for _ in range(3):
        t = plan.take_next(); plan.report(t.index, True)           # warm-up
    start = plan.w
    plan._kinds[99] = "track"
    plan.report(99, True)
    assert plan.w < start
    low = plan.w
    plan.report(99, False)
    assert plan.w > low and plan.step_up > 0.07                    # up-step bigger than down-step


def test_relief_after_two_crashes_and_only_track_moves_staircase():
    plan = _plan(seed=5)
    for _ in range(3):
        t = plan.take_next(); plan.report(t.index, True)
    plan.report(100, False); plan.report(101, False)
    t = plan.take_next()
    assert plan._kinds[t.index] == "relief" and t.target_w >= min(W_MAX, 1.99 * plan.w)
    before = plan.w
    plan.report(t.index, False)                                    # a relief crash must not change it
    assert plan.w == before


def test_staircase_settles_near_the_players_threshold():
    """A simulated player (bias 0, spread 60 ms): the track windows should hover
    near the width that player clears about 75% of the time (about 126 ms)."""
    from math import erf, sqrt
    phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    rng = random.Random(0)
    ws = []
    for seed in range(40):
        plan = _plan(seed, levels=3, per_level=40, start_w=0.32)
        for t in _drain(plan, lambda t: rng.random() < 2 * phi(t.target_w / 2 / 0.06) - 1):
            if plan._kinds[t.index] == "track" and t.index > 40:
                ws.append(t.target_w)
    mean_w = sum(ws) / len(ws)
    assert 0.09 < mean_w < 0.17


def test_plans_with_same_seed_repeat_and_different_seeds_differ():
    a, b, c = (_drain(_plan(s)) for s in (1, 1, 2))
    assert [t.target_w for t in a] == [t.target_w for t in b]
    assert [t.target_w for t in a] != [t.target_w for t in c]
