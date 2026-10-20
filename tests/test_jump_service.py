import math
import random

import pytest

from gesture_lab.adaptive import Config, timing_window, time_above, difficulty_profile
from gesture_lab import jump_service as js

CFG = Config()


def test_kinematics_identities():
    k = js.kinematics(60.0, CFG)
    assert k.t1 + k.t2 == pytest.approx(k.t_air)
    assert k.t2 - k.t1 == pytest.approx(k.t_above)
    assert k.t_above == pytest.approx(time_above(60.0, CFG))
    assert k.r ** 2 == pytest.approx(2 * CFG.GRAVITY * (js.jump_peak(CFG) - 60.0))


def test_above_apex_is_infeasible():
    assert js.kinematics(js.jump_peak(CFG) + 1, CFG) is None
    assert not js.success_window(200, 30, 400, None, CFG).feasible


def test_clearability_matches_closed_form():
    h, w, s = 100.0, 50.0, 550.0
    tau = (w + CFG.PLAYER_WIDTH) / s
    closed = h <= js.jump_peak(CFG) - CFG.GRAVITY * tau ** 2 / 8
    assert js.is_clearable(h, w, s, CFG) == closed


def test_clearance_window_width_matches_repo_timing_window():
    h, w, s = 70.0, 35.0, 405.0
    win = js.success_window(h, w, s, None, CFG)
    assert win.U == pytest.approx(win.U_clear)
    assert win.width == pytest.approx(timing_window(h, w, s, CFG))


def test_gap_limited_width_formula():
    h, w, s, gap = 40.0, 30.0, 400.0, 120.0
    k = js.kinematics(h, CFG)
    win = js.success_window(h, w, s, gap, CFG)
    assert win.U == pytest.approx(win.U_gap)
    assert win.width == pytest.approx((gap - CFG.PLAYER_WIDTH) / s - k.t1)


def test_gap_binds_iff_small_gap():
    h, w, s = 60.0, 30.0, 400.0
    k = js.kinematics(h, CFG)
    threshold = s * k.t2 - w
    assert js.success_window(h, w, s, threshold - 10, CFG).U_gap < js.success_window(h, w, s, threshold - 10, CFG).U_clear
    assert js.success_window(h, w, s, threshold + 10, CFG).U_gap > js.success_window(h, w, s, threshold + 10, CFG).U_clear


def test_pmax_formula_for_centred_gaussian():
    win = js.success_window(60.0, 30.0, 400.0, None, CFG)
    model = js.GaussianTiming(mu=0.0, sigma0=0.04)
    expected = 2 * js._phi(win.width / (2 * 0.04)) - 1
    assert js.p_success(win, model, 400.0) == pytest.approx(expected)


def test_infeasible_window_has_zero_probability():
    win = js.success_window(200.0, 30.0, 400.0, None, CFG)
    assert js.p_success(win, js.GaussianTiming(), 400.0) == 0.0


def test_noisier_player_is_less_likely_to_succeed():
    win = js.success_window(60.0, 30.0, 400.0, None, CFG)
    lo = js.p_success(win, js.GaussianTiming(sigma0=0.03), 400.0)
    hi = js.p_success(win, js.GaussianTiming(sigma0=0.09), 400.0)
    assert lo > hi


def test_speed_dependent_sigma():
    m = js.GaussianTiming(sigma0=0.04, alpha=1.0, s_ref=400.0)
    assert m.sigma(800.0) == pytest.approx(0.08)


def test_fit_gaussian_recovers_parameters():
    rng = random.Random(0)
    xs = [rng.gauss(0.02, 0.05) for _ in range(5000)]
    m = js.fit_gaussian(xs)
    assert m.mu == pytest.approx(0.02, abs=0.005)
    assert m.sigma0 == pytest.approx(0.05, abs=0.005)


def test_empirical_matches_gaussian_on_gaussian_data():
    rng = random.Random(1)
    xs = tuple(rng.gauss(0.0, 0.05) for _ in range(2000))
    emp = js.EmpiricalTiming(xs, bandwidth=0.01)
    gau = js.fit_gaussian(xs)
    assert emp.cdf(0.03) == pytest.approx(gau.cdf(0.03), abs=0.03)


def test_solve_difficulty_orders_by_skill():
    good = js.GaussianTiming(sigma0=0.02)
    poor = js.GaussianTiming(sigma0=0.12)
    d_good = js.solve_difficulty(0.75, good, CFG)
    d_poor = js.solve_difficulty(0.75, poor, CFG)
    assert d_good >= d_poor
    assert js.p_success_at_d(d_poor, poor, CFG) >= 0.75 or d_poor == CFG.D_MIN


def test_observed_delta_zero_at_centre():
    win = js.success_window(60.0, 30.0, 400.0, None, CFG)
    assert js.observed_delta(win.centre, win) == pytest.approx(0.0)


def test_height_for_window_round_trip():
    for target in (0.45, 0.24, 0.09, 0.06):
        h = js.height_for_window(target, 30.0, 400.0, CFG)
        assert js.success_window(h, 30.0, 400.0, None, CFG).width == pytest.approx(target)


def test_height_for_window_rejects_impossible_targets():
    with pytest.raises(ValueError):
        js.height_for_window(0.0, 30.0, 400.0, CFG)
    with pytest.raises(ValueError):
        js.height_for_window(5.0, 30.0, 400.0, CFG)   # wider than any jump allows
