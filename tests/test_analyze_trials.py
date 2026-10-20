"""Tests for analyze_trials.py."""
import io
import contextlib
import random
import statistics

import pytest

from gesture_lab import analyze_trials as at
from gesture_lab import experiment as ex
from gesture_lab import simulate_experiment as sim


def _tagged(rows):
    for r in rows:
        r["_source"] = "test"
    return rows


def _report(rows):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = at.build_report(rows)
    return out, buf.getvalue()


def test_load_trials_skips_comment_lines(tmp_path):
    path = tmp_path / "t.csv"
    sim.write_csv(sim.simulate(seed=1, blocks=1), str(path))
    rows = at.load_trials([str(path)])
    assert len(rows) == 10
    assert set(ex.TRIAL_HEADER) <= set(rows[0])


def test_skewness_and_kurtosis_of_known_samples():
    rng = random.Random(0)
    gauss = [rng.gauss(0, 1) for _ in range(5000)]
    expo = [rng.expovariate(1.0) for _ in range(5000)]
    assert abs(at.skewness(gauss)) < 0.15 and abs(at.excess_kurtosis(gauss)) < 0.3
    assert at.skewness(expo) > 1.5                      # clearly lopsided


def test_qq_correlation_separates_gaussian_from_heavy_tails():
    rng = random.Random(1)
    gauss = [rng.gauss(0, 1) for _ in range(300)]
    heavy = [rng.gauss(0, 1) / max(abs(rng.gauss(0, 1)), 0.05) for _ in range(300)]
    assert at.qq_correlation(gauss) > 0.99
    assert at.qq_correlation(heavy) < at.qq_correlation(gauss) - 0.05


def test_report_recovers_bias_and_spread():
    rows = []
    for seed in range(8):
        rows += sim.simulate(mu=0.03, sigma=0.05, seed=seed)
    out, text = _report(_tagged(rows))
    assert out["mu"] == pytest.approx(0.03 + ex.FRAME_DT / 2, abs=0.01)
    assert out["sigma"] == pytest.approx(0.05, abs=0.01)
    assert out["physics_check"]["n_far_off"] == 0
    assert "good description" in text
    assert "Jump timing report" in text


def test_model_prediction_matches_observed_rates_on_gaussian_player():
    rows = []
    for seed in range(12):
        rows += sim.simulate(mu=0.0, sigma=0.06, seed=seed)
    out, _ = _report(_tagged(rows))
    for row in out["table"]:
        if row["n"] >= 40:
            assert row["rate"] == pytest.approx(row["predicted"], abs=0.12)


def test_narrow_windows_are_harder():
    rows = []
    for seed in range(8):
        rows += sim.simulate(mu=0.0, sigma=0.06, seed=seed)
    out, _ = _report(_tagged(rows))
    rates = [t["rate"] for t in out["table"]]       # widest window first
    assert rates[0] > rates[-1]


def test_too_few_jumps_gives_a_clear_message():
    rows = _tagged(sim.simulate(seed=0, blocks=1))[:5]
    out, text = _report(rows)
    assert "mu" not in out and "Play a longer session" in text


def test_missing_physics_is_flagged():
    rows = _tagged(sim.simulate(seed=0, blocks=2))
    for r in rows:                                   # pretend the game disagrees
        r["outcome"] = "0" if r["outcome"] == "1" else "1"
        r["failure_mode"] = "unexplained"
    _, text = _report(rows)
    assert "MISSING" in text


def test_plot_and_cli(tmp_path):
    path, png = tmp_path / "t.csv", tmp_path / "t.png"
    sim.write_csv(sim.simulate(seed=2, blocks=3), str(path))
    with contextlib.redirect_stdout(io.StringIO()):
        assert at.main([str(path), "--plot", str(png)]) == 0
    assert png.exists() and png.stat().st_size > 5000


def test_timing_uses_delta_open_but_the_outcome_model_keeps_the_game_clock():
    plain = sim.simulate(seed=3, mu=0.03, sigma=0.05, blocks=6)
    shifted = sim.simulate(seed=3, mu=0.03, sigma=0.05, blocks=6)
    for r in shifted:
        if r["delta"] != "":
            r["delta_open"] = f"{float(r['delta']) - 0.05:.4f}"      # camera saw it 50 ms earlier
    out_plain, _ = _report(_tagged(plain))
    out, text = _report(_tagged(shifted))
    assert out["mu"] == pytest.approx(out_plain["mu"] - 0.05, abs=1e-3)       # bias is the corrected one
    assert out["sigma"] == pytest.approx(out_plain["sigma"], abs=1e-3)        # a constant shift keeps the spread
    assert [t["predicted"] for t in out["table"]] == [t["predicted"] for t in out_plain["table"]]
    assert "50 ms" in text and "later on average" in text


def test_old_logs_without_delta_open_still_work():
    rows = _tagged(sim.simulate(seed=4, mu=0.0, sigma=0.05, blocks=4))
    assert at.timing_key(rows) == "delta"
    _, text = _report(rows)
    assert "later on average" not in text
