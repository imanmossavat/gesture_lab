"""Tests for the HTML report: the maths helpers, the drift model, and the page."""
import random

from report_html import build_html, fit_drift, p_clear, window_for_rate, wilson


def _rows(n=24, speeds=(400.0,), seed=1, bias=-20.0, sigma=50.0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        w = rng.choice([0.45, 0.24, 0.13, 0.09])
        delta = rng.gauss(bias, sigma) / 1000.0
        ok = abs(delta) <= w / 2
        rows.append({"player": "p", "_source": "t.csv", "trial": str(i), "W": str(w),
                     "delta": str(delta), "t_j": "1", "outcome": str(int(ok)),
                     "predicted_success": str(int(ok)), "speed": str(speeds[i % len(speeds)])})
    return rows


def test_p_clear_and_window_for_rate_are_inverse():
    assert abs(p_clear(100000, 0, 50) - 1.0) < 1e-9
    assert abs(p_clear(0.001, 0, 50)) < 1e-3
    w = window_for_rate(0.7, -20.0, 60.0)
    assert abs(p_clear(w, -20.0, 60.0) - 0.7) < 1e-6


def test_wilson_range_contains_the_rate_and_is_wide_for_small_n():
    lo, hi = wilson(3, 4)
    assert lo < 0.75 < hi and hi - lo > 0.4
    assert wilson(0, 0) == (0.0, 1.0)


def test_drift_model_is_quiet_when_flat_and_notices_a_big_drift():
    flat = sum(fit_drift([random.Random(s).gauss(-20, 50) for _ in range(30)])["drifting"] for s in range(20))
    rng = random.Random(0)
    moving = [rng.gauss(-80 + 160 * i / 29, 30) for i in range(30)]
    assert flat <= 2
    assert fit_drift(moving)["drifting"] is True


def test_drift_path_has_one_value_per_jump():
    d = fit_drift([float(x) for x in range(-30, 30, 2)])
    assert len(d["mean"]) == len(d["sd"]) == 30


def test_page_is_selfcontained_and_has_the_sections():
    page = build_html(_rows())
    for text in ("Jump timing report", "<svg", "Drift", "Chance of clearing", "Where your jumps landed"):
        assert text in page
    assert "http://" not in page.replace("http://www.w3.org", "") and "<script" not in page


def test_by_speed_table_only_appears_with_several_speeds():
    assert "By speed" not in build_html(_rows())
    assert "By speed" in build_html(_rows(n=40, speeds=(400.0, 460.0)))


def test_short_session_gets_a_polite_message_not_a_crash():
    assert "Fewer than 8" in build_html(_rows(n=5))


def test_page_explains_the_two_clocks_when_delta_open_is_present():
    rows = _rows()
    for r in rows:
        r["delta_open"] = str(float(r["delta"]) - 0.05)
    page = build_html(rows)
    assert "later on average" in page and "50 ms later" in page
    assert "later on average" not in build_html(_rows())
