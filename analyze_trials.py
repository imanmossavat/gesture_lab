#!/usr/bin/env python3
"""Analyze a trial log: how precisely does this player time their jump?

Input : one or more CSV files written by ``game.py --log-trials`` (or by
        ``simulate_experiment.py``).
Output: a short written report in plain language, and optionally a picture.

    python analyze_trials.py trials.csv
    python analyze_trials.py trials.csv --plot report.png
    python analyze_trials.py a.csv b.csv          # pool several sessions

How to read the report
----------------------
Every obstacle has a *safe window*: the span of jump times that clear it.  The
physics gives its width ``W`` (in milliseconds).  A narrow window needs a more
precise jump.  For every obstacle we know

    t_j    when the player jumped
    delta  how far t_j was from the CENTRE of the safe window
           (0 = perfect, negative = early, positive = late)

A player's timing is summarized by two numbers:

    bias    the average delta     -> "always a bit late/early"
    spread  the standard deviation of delta  -> "how scattered the jumps are"

If delta follows a bell curve (a Gaussian) with that bias and spread, we can
predict the chance of clearing an obstacle of ANY window width:

    P(clear) = P( -W/2 <= delta <= +W/2 )

The report checks that prediction against what actually happened, and checks
whether the bell-curve assumption is reasonable.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Optional, Sequence

from analyze import wilson          # 95 % confidence interval for a proportion
import jump_service as js           # GaussianTiming: the bell-curve model

MIN_TRIALS_FOR_FIT = 8              # below this, numbers are too uncertain to print
FRAME_MS = 1000.0 / 60.0            # one game frame, used as the physics tolerance


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_trials(paths: Sequence[str]) -> list[dict]:
    """Read trial CSVs.  Lines starting with ``#`` are metadata and are skipped."""
    rows: list[dict] = []
    for path in paths:
        with open(path, newline="") as f:
            data_lines = (line for line in f if not line.startswith("#"))
            for row in csv.DictReader(data_lines):
                row["_source"] = Path(path).name
                rows.append(row)
    return rows


def _flag(row: dict, key: str) -> bool:
    """True when a 0/1 column equals 1 (works for CSV text and for numbers)."""
    value = row.get(key, "")
    return value not in ("", None) and float(value) == 1.0


def _num(row: dict, key: str) -> Optional[float]:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


# ---------------------------------------------------------------------------
# Small statistics helpers (plain Python, so each step can be read)
# ---------------------------------------------------------------------------

def timing_key(rows: Sequence[dict]) -> str:
    """TIMING: the column holding the player's timing.

    ``delta_open`` (when the camera first SAW the gesture, minus half a camera frame)
    if the log has it, otherwise ``delta`` (when the game started the jump).  The
    outcome model in section 4 always uses ``delta``: that is what decides clear or crash.
    """
    return "delta_open" if any(_num(r, "delta_open") is not None for r in rows) else "delta"


def mean_and_spread(values: Sequence[float]) -> tuple[float, float]:
    """Average and sample standard deviation."""
    mu = statistics.fmean(values)
    sigma = statistics.stdev(values) if len(values) > 1 else float("nan")
    return mu, sigma


def skewness(values: Sequence[float]) -> float:
    """0 for a symmetric distribution; positive = long tail to the right."""
    n = len(values)
    mu, sd = statistics.fmean(values), statistics.pstdev(values)
    if n < 3 or sd == 0:
        return float("nan")
    return sum(((x - mu) / sd) ** 3 for x in values) / n


def excess_kurtosis(values: Sequence[float]) -> float:
    """0 for a Gaussian; positive = heavier tails (more extreme jumps)."""
    n = len(values)
    mu, sd = statistics.fmean(values), statistics.pstdev(values)
    if n < 4 or sd == 0:
        return float("nan")
    return sum(((x - mu) / sd) ** 4 for x in values) / n - 3.0


def qq_points(values: Sequence[float]) -> tuple[list[float], list[float]]:
    """Points for a QQ plot of the values against a standard bell curve.

    Sort the values; give the i-th smallest the 'expected' position it would
    have if the data were Gaussian; if the data are Gaussian the points fall on
    a straight line.
    """
    mu, sd = mean_and_spread(values)
    n = len(values)
    normal = statistics.NormalDist()
    theory = [normal.inv_cdf((i + 0.5) / n) for i in range(n)]
    sample = [(x - mu) / sd for x in sorted(values)]
    return theory, sample


def qq_correlation(values: Sequence[float]) -> float:
    """Straightness of the QQ plot: 1.0 = perfectly Gaussian-looking."""
    theory, sample = qq_points(values)
    return statistics.correlation(theory, sample)


# ---------------------------------------------------------------------------
# The analyses
# ---------------------------------------------------------------------------

def physics_check(rows: Sequence[dict]) -> dict:
    """Does the physics window predict what the game decided?

    For every obstacle where the player jumped, compare
        predicted_success  (was t_j inside [L, U]?)  with  outcome.
    Disagreements within one frame of a window edge are expected: the game
    updates in 1/60 s steps.  Anything further away means the physics model is
    wrong for this game.
    """
    jumped = [r for r in rows if _flag(r, "jumped")]
    agree = [r for r in jumped if _flag(r, "predicted_success") == _flag(r, "outcome")]
    far_off = []
    for r in jumped:
        if _flag(r, "predicted_success") != _flag(r, "outcome"):
            t_j, lo, hi = _num(r, "t_j"), _num(r, "L"), _num(r, "U")
            distance_ms = min(abs(t_j - lo), abs(t_j - hi)) * 1000.0
            if distance_ms > FRAME_MS:
                far_off.append(r)
    unexplained = [r for r in rows if r["failure_mode"] == "unexplained"]
    return {
        "n_jumped": len(jumped), "n_agree": len(agree),
        "n_far_off": len(far_off), "n_unexplained": len(unexplained),
    }


def success_by_window(rows: Sequence[dict], fit: Optional[js.GaussianTiming]) -> list[dict]:
    """Success rate for each group of similar window widths.

    Obstacles are grouped by the window width they were designed for
    (``target_w``).  If that column is empty, windows are rounded to 20 ms.
    """
    groups: dict[float, list[dict]] = {}
    for r in rows:
        w = _num(r, "W")
        if w is None:
            continue
        key = _num(r, "target_w")
        key = key if key is not None else round(w / 0.02) * 0.02
        groups.setdefault(key, []).append(r)

    table = []
    for key in sorted(groups, reverse=True):
        group = groups[key]
        n = len(group)
        k = sum(int(float(r["outcome"])) for r in group)
        mean_w = statistics.fmean(_num(r, "W") for r in group)
        lo, hi = wilson(k, n)
        predicted = None
        if fit is not None:
            half = mean_w / 2
            predicted = fit.cdf(half) - fit.cdf(-half)
        table.append({
            "target_w": key, "mean_w": mean_w, "n": n, "k": k,
            "rate": k / n, "lo": lo, "hi": hi, "predicted": predicted,
        })
    return table


def spread_by_window(rows: Sequence[dict], key: str = "delta") -> list[tuple[float, int, float]]:
    """Spread of delta for wide / medium / narrow windows.

    A simple bell-curve model assumes the spread is the same for every
    obstacle.  If narrow windows show a clearly different spread, the model
    needs to depend on the window (or the height).
    """
    pairs = [(_num(r, "W"), _num(r, key)) for r in rows]
    pairs = sorted((w, d) for w, d in pairs if w is not None and d is not None)
    if len(pairs) < 9:
        return []
    third = len(pairs) // 3
    chunks = [pairs[:third], pairs[third:2 * third], pairs[2 * third:]]
    out = []
    for chunk in chunks:
        deltas = [d for _, d in chunk]
        out.append((statistics.fmean(w for w, _ in chunk), len(chunk),
                    statistics.stdev(deltas)))
    return out


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def _ms(seconds: float) -> str:
    return f"{seconds * 1000:+.0f} ms"


def _heading(number: int, title: str) -> None:
    print(f"\n{number}. {title}")
    print("-" * (len(title) + 4))


def build_report(rows: Sequence[dict]) -> dict:
    """Print the report and return the numbers (so tests and plots can use them)."""
    out: dict = {}
    n = len(rows)
    if n == 0:
        print("No obstacles found in the input.")
        return out

    outcomes = [int(float(r["outcome"])) for r in rows]
    dkey = timing_key(rows)
    deltas = [d for d in (_num(r, dkey) for r in rows) if d is not None]          # your timing
    game_deltas = [d for d in (_num(r, "delta") for r in rows) if d is not None]  # decides the outcome
    sources = sorted({r["_source"] for r in rows})

    print("=" * 64)
    print(" Jump timing report")
    print("=" * 64)
    print(f"Files: {', '.join(sources)}")

    # --- 1 -----------------------------------------------------------
    _heading(1, "What was recorded")
    cleared = sum(outcomes)
    print(f"{n} obstacles: {cleared} cleared, {n - cleared} crashed "
          f"({cleared / n:.0%} cleared).")
    modes = {m: sum(1 for r in rows if r["failure_mode"] == m)
             for m in ("early", "late", "no_jump", "unexplained")}
    print("Why the crashes happened:")
    print(f"  jumped too early       : {modes['early']}")
    print(f"  jumped too late        : {modes['late']}")
    print(f"  no jump in time        : {modes['no_jump']}")
    print(f"  not explained by physics: {modes['unexplained']}  "
          "(a few right at a window edge are normal)")
    print("What this means: 'too early' = the jump started before the safe window\n"
          "and the player came down on the obstacle; 'too late' = the jump started\n"
          "after the window and the player hit the obstacle on the way up.")
    out["failure_modes"] = modes

    # --- 2 -----------------------------------------------------------
    _heading(2, "Does the physics predict the game?")
    check = physics_check(rows)
    out["physics_check"] = check
    if check["n_jumped"] == 0:
        print("No jumps were recorded, so there is nothing to compare.")
    else:
        print(f"Of {check['n_jumped']} jumps, the physics window predicted the game's "
              f"result for {check['n_agree']}.")
        print(f"Disagreements more than one frame ({FRAME_MS:.0f} ms) from a window edge: "
              f"{check['n_far_off']}.")
        if check["n_far_off"] == 0:
            print("What this means: the physics window is a good description of this\n"
                  "game. Differences are only at the very edges of a window, where one\n"
                  "frame decides.")
        else:
            print("What this means: the physics window is MISSING something. Look at\n"
                  "the rows with failure_mode 'unexplained' before trusting the\n"
                  "timing analysis below. Likely causes: wrong player or obstacle\n"
                  "sizes, a changed jump speed, or a frame rate far from 60.")

    # --- 3 -----------------------------------------------------------
    _heading(3, "Your timing: bias and spread")
    fit = None
    if len(deltas) < MIN_TRIALS_FOR_FIT:
        print(f"Only {len(deltas)} jumps recorded; at least {MIN_TRIALS_FOR_FIT} are "
              "needed for a meaningful estimate. Play a longer session.")
    else:
        mu, sigma = mean_and_spread(deltas)
        # Standard error of the average, and an approximate one for the spread.
        se_mu = sigma / math.sqrt(len(deltas))
        se_sigma = sigma / math.sqrt(2 * (len(deltas) - 1))
        g_mu, g_sigma = mean_and_spread(game_deltas)
        fit = js.GaussianTiming(mu=g_mu, sigma0=g_sigma)     # predicts outcomes, so it uses the game clock
        out.update(mu=mu, sigma=sigma, se_mu=se_mu, se_sigma=se_sigma, n_deltas=len(deltas))
        direction = "late" if mu > 0 else "early"
        print(f"Bias  : {_ms(mu)}  (about +-{se_mu * 1000:.0f} ms uncertainty)")
        print(f"        On average you jump {abs(mu) * 1000:.0f} ms {direction} compared with "
              "the middle of the safe window.")
        print(f"Spread: {sigma * 1000:.0f} ms  (about +-{se_sigma * 1000:.0f} ms uncertainty)")
        print("        Roughly 2 out of 3 of your jumps land within this distance of your\n"
              "        own average.")
        print(f"Based on {len(deltas)} jumps.")
        if dkey == "delta_open":
            pairs = [(_num(r, "delta"), _num(r, "delta_open")) for r in rows]
            shift = statistics.fmean(a - b for a, b in pairs if a is not None and b is not None)
            print("Timing is when the camera first saw your gesture, minus half a camera frame,\n"
                  "so the game's own wait is removed. The game then started your jumps\n"
                  f"{shift * 1000:.0f} ms later on average, and that later time decides clear or crash\n"
                  "(section 4 uses it).")
        missing = n - len(deltas)
        if missing:
            print(f"Note: {missing} obstacles had no jump before they arrived. They are\n"
                  "left out of bias and spread, so these numbers are a little too\n"
                  "optimistic (a player who never jumps in time has no timing to measure).")

    # --- 4 -----------------------------------------------------------
    _heading(4, "Success against window width")
    table = success_by_window(rows, fit)
    out["table"] = table
    header = f"{'window':>9} {'obstacles':>10} {'cleared':>8} {'rate':>6} {'95% range':>12}"
    if fit is not None:
        header += f" {'model says':>11}"
    print(header)
    for t in table:
        line = (f"{t['mean_w'] * 1000:6.0f} ms {t['n']:10d} {t['k']:8d} "
                f"{t['rate']:6.0%} {t['lo']:5.0%}-{t['hi']:<5.0%}")
        if t["predicted"] is not None:
            line += f" {t['predicted']:10.0%}"
        print(line)
    print("What this means: wider windows are easier. The '95% range' shows how much\n"
          "luck could explain with this few obstacles; short sessions give wide ranges.")
    if fit is not None:
        print("'model says' is the chance predicted from your bias and spread alone.\n"
              "If the observed rate is often far outside the 95% range of the\n"
              "prediction, the bell-curve model is not describing you well.")

    # --- 5 -----------------------------------------------------------
    _heading(5, "Is a bell curve a reasonable model?")
    if len(deltas) < MIN_TRIALS_FOR_FIT:
        print("Not enough jumps to check.")
    else:
        skew = skewness(deltas)
        kurt = excess_kurtosis(deltas)
        straight = qq_correlation(deltas)
        out.update(skew=skew, kurtosis=kurt, qq_r=straight)
        print(f"Lopsidedness (skewness)     : {skew:+.2f}   (0 = symmetric)")
        print(f"Extreme jumps (kurtosis)    : {kurt:+.2f}   (0 = bell curve, higher = more outliers)")
        print(f"Straightness of the QQ plot : {straight:.3f}   (1 = bell curve)")
        concerns = []
        if abs(skew) > 0.5:
            concerns.append("jump errors lean to one side")
        if kurt > 1.0:
            concerns.append("there are more extreme jumps than a bell curve allows")
        if straight < 0.97:
            concerns.append("the QQ plot is visibly bent")
        if len(deltas) < 30:
            print("With fewer than 30 jumps these checks are rough.")
        if concerns:
            print("What this means: the bell curve is questionable here: "
                  + "; ".join(concerns) + ".")
        else:
            print("What this means: nothing here contradicts a bell curve, within what\n"
                  "this many jumps can show.")

        thirds = spread_by_window(rows, dkey)
        out["spread_by_window"] = thirds
        if thirds:
            print("\nIs the spread the same for wide and narrow windows?")
            for mean_w, count, sd in thirds:
                print(f"  windows around {mean_w * 1000:4.0f} ms ({count:2d} jumps): "
                      f"spread {sd * 1000:3.0f} ms")
            print("What this means: a plain bell curve assumes these three numbers are\n"
                  "alike. If they differ a lot, tall (narrow-window) obstacles make the\n"
                  "player's timing less precise, and the model should depend on height.")

        half = len(deltas) // 2
        if half >= 4:
            first, second = deltas[:half], deltas[half:]
            print("\nDoes timing change during the session?")
            print(f"  first half : bias {_ms(statistics.fmean(first))}, "
                  f"spread {statistics.stdev(first) * 1000:.0f} ms")
            print(f"  second half: bias {_ms(statistics.fmean(second))}, "
                  f"spread {statistics.stdev(second) * 1000:.0f} ms")
            # Is the difference in bias bigger than chance would usually give?
            se = math.sqrt(statistics.variance(first) / len(first)
                           + statistics.variance(second) / len(second))
            diff = statistics.fmean(second) - statistics.fmean(first)
            out["half_diff"], out["half_diff_se"] = diff, se
            if abs(diff) > 2 * se:
                print("What this means: the bias changed by "
                      f"{abs(diff) * 1000:.0f} ms, more than chance usually explains\n"
                      "(about 2 x " f"{se * 1000:.0f} ms). The player was warming up or tiring,\n"
                      "so one pooled bias and spread is too simple.")
            else:
                print("What this means: the change between halves is "
                      f"{abs(diff) * 1000:.0f} ms, which is within what chance could\n"
                      f"produce (about 2 x {se * 1000:.0f} ms). No clear warm-up or tiring effect.")
    print()
    return out


# ---------------------------------------------------------------------------
# The picture
# ---------------------------------------------------------------------------

def make_plot(rows: Sequence[dict], summary: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deltas = [d for d in (_num(r, timing_key(rows)) for r in rows) if d is not None]
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Jump timing: what happened, and whether the model fits", fontsize=14)

    fit = None
    if "mu" in summary:
        fit = js.GaussianTiming(mu=summary["mu"], sigma0=summary["sigma"])

    # (a) success against window width ------------------------------------
    a = ax[0][0]
    table = summary.get("table", [])
    x = [t["mean_w"] * 1000 for t in table]
    y = [t["rate"] for t in table]
    err = [[max(0.0, t["rate"] - t["lo"]) for t in table],
           [max(0.0, t["hi"] - t["rate"]) for t in table]]
    a.errorbar(x, y, yerr=err, fmt="o", color="#2b6cb0", capsize=3, label="observed (95% range)")
    if fit is not None and table:
        grid = [i * (max(x) * 1.1) / 100 for i in range(1, 101)]
        a.plot(grid, [fit.cdf(w / 2000) - fit.cdf(-w / 2000) for w in grid],
               color="#c0392b", label="model from bias and spread")
    a.set_ylim(-0.05, 1.05)
    a.set_xlabel("width of the safe window (ms)")
    a.set_ylabel("share of obstacles cleared")
    a.set_title("Narrower windows are harder")
    a.legend(loc="lower right")

    # (b) where jumps landed -----------------------------------------------
    b = ax[0][1]
    if deltas:
        counts, edges, _ = b.hist([d * 1000 for d in deltas], bins=12,
                                  color="#9ec5e8", edgecolor="white")
        if fit is not None:
            # Scale the bell curve so it is comparable with the bar heights.
            bin_width = edges[1] - edges[0]
            curve = statistics.NormalDist(fit.mu * 1000, fit.sigma0 * 1000)
            lo, hi = edges[0] - 50, edges[-1] + 50
            pts = [lo + (hi - lo) * i / 200 for i in range(201)]
            b.plot(pts, [len(deltas) * bin_width * curve.pdf(v) for v in pts],
                   color="#c0392b", label="fitted bell curve")
            b.legend()
    b.axvline(0, color="k", lw=1)
    b.set_xlabel("jump time minus centre of safe window (ms)   <- early | late ->")
    b.set_ylabel("number of jumps")
    b.set_title("Where your jumps landed (0 = perfect)")

    # (c) QQ plot --------------------------------------------------------
    c = ax[1][0]
    if len(deltas) >= 4:
        theory, sample = qq_points(deltas)
        c.plot(theory, sample, "o", color="#2b6cb0")
        lim = max(abs(v) for v in theory + sample) * 1.1
        c.plot([-lim, lim], [-lim, lim], color="#c0392b", lw=1, label="perfect bell curve")
        c.legend()
    c.set_xlabel("expected position if bell-shaped")
    c.set_ylabel("actual position (in spreads)")
    c.set_title("Straight line = bell curve is reasonable")

    # (d) every obstacle in order --------------------------------------------
    d = ax[1][1]
    for r in rows:
        dl, w = _num(r, "delta"), _num(r, "W")
        if dl is None or w is None:
            continue
        i = int(r["trial"])
        ok = _flag(r, "outcome")
        d.plot([i, i], [-w * 500, w * 500], color="#bbbbbb", lw=3, solid_capstyle="butt", zorder=1)
        d.plot(i, dl * 1000, "o", color="#2f9e44" if ok else "#c0392b", zorder=2)
    d.axhline(0, color="k", lw=1)
    d.set_xlabel("obstacle number")
    d.set_ylabel("timing error (ms)")
    d.set_title("Each jump against its safe zone (grey bar)")
    d.text(0.01, 0.02, "green = cleared, red = crashed", transform=d.transAxes, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze jump-timing trial logs.")
    parser.add_argument("files", nargs="+", help="trial CSV file(s)")
    parser.add_argument("--plot", metavar="PNG", help="also save the four-panel picture")
    parser.add_argument("--html", metavar="HTML", help="also save a self-contained HTML report (adds a drift model)")
    args = parser.parse_args(argv)

    rows = load_trials(args.files)
    summary = build_report(rows)
    if args.plot and rows:
        make_plot(rows, summary, args.plot)
        print(f"Picture saved to {args.plot}")
    if args.html and rows:
        from report_html import build_html   # imported here: report_html imports this module
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(build_html(rows))
        print(f"HTML report saved to {args.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
