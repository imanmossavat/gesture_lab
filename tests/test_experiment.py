"""Tests for experiment.py: the obstacle plan, the recorder, and the physics
correction for the game's frame-by-frame update."""
import random
import statistics

import pytest

from gesture_lab.adaptive import Config, Obstacle
from gesture_lab import experiment as ex
from gesture_lab import jump_service as js
from gesture_lab import simulate_experiment as sim

CFG = Config()
FRONT = CFG.PLAYER_X + CFG.PLAYER_WIDTH


# ---------------------------------------------------------------- physics
def test_frame_by_frame_apex_matches_game_physics():
    """Step the jump the way game.py does and compare with the corrected physics."""
    dt, g, v = ex.FRAME_DT, CFG.GRAVITY, -CFG.JUMP_VELOCITY
    y = peak = 0.0
    for _ in range(200):
        v -= g * dt
        y += v * dt
        peak = max(peak, y)
    assert peak == pytest.approx(js.jump_peak(ex.game_physics(CFG)), abs=0.5)
    assert peak < js.jump_peak(CFG)          # the textbook apex is too optimistic


def test_game_window_is_shifted_one_frame_later():
    textbook = js.success_window(60.0, 30.0, 400.0, None, ex.game_physics(CFG))
    game = ex.game_window(60.0, 30.0, 400.0, None, CFG)
    assert game.L == pytest.approx(textbook.L + ex.FRAME_DT)
    assert game.width == pytest.approx(textbook.width)


# ------------------------------------------------------------------- plan
def _trials(seed=0, **kw):
    return ex.make_trials(CFG, random.Random(seed), **kw)


def test_plan_length_and_warmup():
    trials = _trials(blocks=4, warmup=3)
    assert len(trials) == 3 + 7 * 4
    assert all(t.target_w == max(ex.DEFAULT_WINDOWS) for t in trials[:3])


def test_each_block_uses_every_window_once():
    trials = _trials(blocks=4, warmup=3)
    body = [t.target_w for t in trials[3:]]
    for i in range(4):
        assert sorted(body[i * 7:(i + 1) * 7]) == sorted(ex.DEFAULT_WINDOWS)


def test_order_is_safe_and_not_repetitive():
    for seed in range(20):
        body = [t.target_w for t in _trials(seed)[3:]]
        assert ex._is_safe_order(body)
        assert all(a != b for a, b in zip(body, body[1:]))


def test_plan_is_reproducible_and_varies_with_seed():
    assert _trials(1) == _trials(1)
    assert [t.height for t in _trials(1)] != [t.height for t in _trials(2)]


def test_every_obstacle_is_feasible_and_gap_never_binds():
    for seed in range(10):
        for t in _trials(seed):
            win = ex.game_window(t.height, t.width, t.speed, t.gap_after_px, CFG)
            assert win.feasible and win.width > 0.04
            assert win.U_clear <= win.U_gap          # the clearance bound is the binding one


def test_spacing_and_last_trial():
    trials = _trials(0)
    assert trials[-1].interval_after is None and trials[-1].gap_after_px is None
    for t in trials[:-1]:
        assert 1.8 <= t.interval_after <= 2.6
        assert t.gap_after_px == pytest.approx(t.speed * t.interval_after - t.width)


def test_plan_hands_out_trials_in_order():
    plan = ex.ExperimentPlan(_trials(0, blocks=1))
    seen = []
    while not plan.exhausted:
        seen.append(plan.take_next().index)
    assert seen == list(range(plan.total)) and plan.take_next() is None


# --------------------------------------------------------------- recorder
def _ob(distance_px, h=60.0, w=30.0, speed=400.0):
    return Obstacle(x=FRONT + distance_px, w=w, h=h, d_spawn=0.0, speed_spawn=speed)


def test_note_jump_measures_time_before_arrival():
    rec = ex.TrialRecorder(CFG)
    ob = _ob(200.0)                    # 200 px away at 400 px/s = 0.5 s before t = 0
    rec.register(ob)
    assert rec.note_jump(400.0) == pytest.approx(-0.5)


def test_later_jump_replaces_earlier_and_nearest_obstacle_wins():
    rec = ex.TrialRecorder(CFG)
    near, far = _ob(200.0), _ob(900.0)
    rec.register(near); rec.register(far)
    rec.note_jump(400.0)
    near.x -= 40.0                      # 0.1 s later
    assert rec.note_jump(400.0) == pytest.approx(-0.4)
    row = rec.finish(near, 1, 1.0)
    assert float(row["t_j"]) == pytest.approx(-0.4)
    assert rec.finish(far, 0, 2.0)["jumped"] == 0


def test_jump_after_obstacle_reached_player_is_not_recorded():
    rec = ex.TrialRecorder(CFG)
    ob = _ob(-1.0)
    rec.register(ob)
    assert rec.note_jump(400.0) is None
    assert rec.finish(ob, 0, 1.0)["failure_mode"] == "no_jump"


def test_finish_row_matches_header_and_physics():
    rec = ex.TrialRecorder(CFG)
    ob = _ob(300.0, h=60.0)
    rec.register(ob, target_w=0.3, next_gap_px=800.0, index=7)
    win = ex.game_window(60.0, 30.0, 400.0, 800.0, CFG)
    ob.x = FRONT - win.centre * 400.0   # place the obstacle so the jump is at the centre
    rec.note_jump(400.0)
    row = rec.finish(ob, 1, 12.5, player="p", session=3)
    assert set(row) == set(ex.TRIAL_HEADER)
    assert len(ex.row_values(row)) == len(ex.TRIAL_HEADER)
    assert row["trial"] == 7 and row["player"] == "p" and row["session"] == 3
    assert float(row["delta"]) == pytest.approx(0.0, abs=1e-3)
    assert row["predicted_success"] == 1 and row["failure_mode"] == ""
    assert row["binding"] == "clearance"


def test_finish_unknown_obstacle_returns_none():
    assert ex.TrialRecorder(CFG).finish(_ob(100.0), 1, 0.0) is None


def test_classify_failure():
    win = js.JumpWindow(L=-0.5, U_clear=-0.2, U_gap=float("inf"), U=-0.2, feasible=True)
    assert ex.classify_failure(1, -0.3, win) == ""
    assert ex.classify_failure(0, None, win) == "no_jump"
    assert ex.classify_failure(0, -0.6, win) == "early"
    assert ex.classify_failure(0, -0.1, win) == "late"
    assert ex.classify_failure(0, -0.3, win) == "unexplained"


# --------------------------------------------- the whole chain, simulated
def test_simulated_game_agrees_with_physics_window():
    """Run the real obstacle logic with a simulated player on several seeds.

    The physics window must predict the game's decision, except within one
    frame of a window edge, and there must be no 'unexplained' failures.
    """
    total = 0
    for seed in range(6):
        rows = sim.simulate(mu=0.02, sigma=0.05, seed=seed)
        total += len(rows)
        for r in rows:
            assert r["failure_mode"] != "unexplained"
            if r["predicted_success"] != r["outcome"]:
                t_j, lo, hi = float(r["t_j"]), float(r["L"]), float(r["U"])
                assert min(abs(t_j - lo), abs(t_j - hi)) <= ex.FRAME_DT + 1e-3
    assert total >= 6 * 31


def test_simulated_timing_is_recovered():
    """The measured bias and spread should match the simulated player.

    The measured bias is about half a frame (8 ms) later than the aimed one,
    because the jump can only start on a frame boundary.
    """
    deltas = []
    for seed in range(10):
        for r in sim.simulate(mu=0.03, sigma=0.05, seed=seed):
            if r["delta"] != "":
                deltas.append(float(r["delta"]))
    assert statistics.fmean(deltas) == pytest.approx(0.03 + ex.FRAME_DT / 2, abs=0.01)
    assert statistics.stdev(deltas) == pytest.approx(0.05, abs=0.01)


def test_gap_to_next_obstacle_is_filled_in_when_it_spawns():
    """Adaptive modes do not plan gaps; the recorder learns them at spawn time."""
    rec = ex.TrialRecorder(CFG)
    first = _ob(300.0, w=30.0)                       # right edge at FRONT + 330
    second = Obstacle(x=FRONT + 330.0 + 120.0, w=30.0, h=60.0,
                      d_spawn=0.0, speed_spawn=400.0)  # 120 px behind it
    rec.register(first)
    rec.register(second)
    row = rec.finish(first, 1, 1.0)
    assert float(row["next_gap_px"]) == pytest.approx(120.0)


def test_planned_gap_is_not_overwritten():
    rec = ex.TrialRecorder(CFG)
    first, second = _ob(300.0), _ob(900.0)
    rec.register(first, next_gap_px=555.0)
    rec.register(second)
    assert float(rec.finish(first, 1, 1.0)["next_gap_px"]) == pytest.approx(555.0)


def test_t_open_removes_lag_and_half_a_camera_frame():
    rec = ex.TrialRecorder(CFG)
    ob = _ob(300.0, h=60.0)
    rec.register(ob, target_w=0.3, next_gap_px=800.0, index=0)
    rec.note_jump(400.0, lag_s=0.033, cam_interval_s=0.0333)
    row = rec.finish(ob, 1, 1.0)
    assert float(row["open_lag_ms"]) == pytest.approx(33.0, abs=0.1)
    assert float(row["cam_interval_ms"]) == pytest.approx(33.3, abs=0.1)
    shift = 0.033 + 0.0333 / 2.0
    assert float(row["t_j"]) - float(row["t_open"]) == pytest.approx(shift, abs=2e-4)
    assert float(row["delta"]) - float(row["delta_open"]) == pytest.approx(shift, abs=2e-4)


def test_open_columns_are_blank_without_lag_info_or_without_a_jump():
    rec = ex.TrialRecorder(CFG)
    a, b = _ob(300.0, h=60.0), _ob(900.0, h=60.0)
    rec.register(a, target_w=0.3, next_gap_px=800.0, index=0)
    rec.register(b, target_w=0.3, next_gap_px=800.0, index=1)
    rec.note_jump(400.0)                                    # old-style call: no timing info
    row = rec.finish(a, 1, 1.0)
    assert row["t_open"] == "" and row["delta_open"] == "" and row["delta"] != ""
    row = rec.finish(b, 0, 2.0)                             # never jumped
    assert row["t_open"] == "" and row["open_lag_ms"] == ""
