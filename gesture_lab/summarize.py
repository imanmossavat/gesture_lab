#!/usr/bin/env python3
"""summarize.py - compact, shareable summary of gesture_lab logs.

Run it locally on your CSVs. It prints one plain-text block with aggregate
numbers only (no raw rows), so you can paste the block into a chat instead of
sending the data files.

Put this file in the repo root (next to adaptive.py), then:

    python -m gesture_lab.summarize data/*.csv
    python -m gesture_lab.summarize data/*.csv --trim-start 10 --anon --out summary.txt

Handles both log types (detected from the header):
  outcomes : t_s, player, session, mode, d_spawn, d_after, speed, height,
             width, outcome            (written by --log-outcomes)
  latency  : frame_read_to_jump_ms, first_open_to_jump_ms, inference_ms,
             loop_ms, accepted         (written by --log-latency)

For outcome logs it also fits a timing-noise model:
    P(clear) = 2*Phi(B/sigma) - 1 = erf(B / (sigma*sqrt(2)))
where B is the half timing window of each obstacle, computed from the logged
speed/height/width with the game's own physics. sigma (ms) is estimated by
maximum likelihood with a 95% profile-likelihood interval.

Assumptions: timing error is normal and unbiased, and timing is the only way to
fail. Missed gestures make sigma look larger than it is.
"""
import argparse
import csv
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gesture_lab.adaptive import Config, air_time, timing_window  # noqa: E402

CFG = Config()
BAND = 0.15          # "settled" = d stays within +-BAND of its late-session mean
TAIL = 30            # obstacles used for late-session d
MIN_BIN_N = 10       # bins with fewer obstacles are flagged "few"
SQRT2 = math.sqrt(2.0)
LAT_COLS = ["frame_read_to_jump_ms", "first_open_to_jump_ms", "inference_ms", "loop_ms"]


def _target_rate(value):
    v = float(value)
    if not 0.0 < v < 1.0:
        raise argparse.ArgumentTypeError(f"target success rate must be in (0, 1), got {v}")
    return v


# ----------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def percentile(vals, q):
    s = sorted(vals)
    k = q * (len(s) - 1)
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] * (c - k) + s[c] * (k - f)


def git_hash():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load(path):
    meta = {}
    with open(path, newline="") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if line.startswith("#"):
                body = line[1:].strip()
                if "=" in body:
                    k, v = body.split("=", 1)
                    meta[k.strip()] = v.strip()
                continue
            f.seek(pos)
            break
        rows = list(csv.DictReader(f))
    return meta, rows


def kind_of(rows):
    cols = set(rows[0]) if rows else set()
    if "outcome" in cols:
        return "outcomes"
    if "first_open_to_jump_ms" in cols:
        return "latency"
    return None


def _meta_line(meta):
    if not meta:
        return ""
    parts = []
    for k in ("ADAPTATION_RATE", "MOMENTUM", "TARGET_SUCCESS_RATE", "D_START", "effective_start_d",
              "SPAWN_GAP_MODE", "SPAWN_REST_LO_S", "SPAWN_REST_HI_S",
              "initial_preset", "player", "timestamp", "git_hash"):
        if k in meta:
            parts.append(f"{k}={meta[k]}")
    return "  [" + ", ".join(parts) + "]"


# ------------------------------------------------------------- sigma fit
def budget_s(row):
    """Half timing window (seconds) for one logged obstacle."""
    h = float(row["height"])
    w = float(row["width"])
    s = float(row["speed"])
    return 0.5 * timing_window(h, w, s, CFG)


def p_clear(b, sigma):
    """P(|error| <= B) for normal error with spread sigma (both in seconds)."""
    if b <= 0.0:
        return 0.0
    return math.erf(b / (sigma * SQRT2))


def loglik(sigma, budgets, outs):
    eps = 1e-12
    total = 0.0
    for b, o in zip(budgets, outs):
        p = min(1.0 - eps, max(eps, p_clear(b, sigma)))
        total += math.log(p) if o else math.log(1.0 - p)
    return total


def fit_sigma(budgets, outs, lo_ms=5, hi_ms=800):
    """Maximum-likelihood sigma on a 1 ms grid plus a 95% profile interval."""
    grid = list(range(lo_ms, hi_ms + 1))
    ll = [loglik(g / 1000.0, budgets, outs) for g in grid]
    best_i = max(range(len(grid)), key=lambda i: ll[i])
    best = ll[best_i]
    inside = [grid[i] for i in range(len(grid)) if ll[i] >= best - 1.92]
    return grid[best_i], min(inside), max(inside), lo_ms, hi_ms


# ---------------------------------------------------------- outcome report
REST_EDGES = [-1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 99.0]


def _rest_label(lo, hi):
    if lo <= -1.0:
        return f"< {hi:.1f}"
    if hi >= 99.0:
        return f">= {lo:.1f}"
    return f"{lo:.1f}-{hi:.1f}"


def rest_table(kept, outs, budgets, sigma_ms):
    """Success of each obstacle against its REST time (needs the gap_s column).

    REST = gap_s - air_time: the time between landing from the previous jump and
    the ideal take-off for this obstacle, if the previous jump was perfectly
    timed.  Only obstacles whose previous obstacle was cleared are used (a crash
    leaves the player in an unknown state).  'model' is the success predicted by
    timing noise alone, so a bin far below it shows recovery/decision time, not
    timing precision, is the limit.
    """
    air = air_time(CFG)
    pts = []  # (rest, outcome, budget)
    for i in range(1, len(kept)):
        try:
            gap = float(kept[i].get("gap_s") or 0.0)
        except ValueError:
            continue
        if gap <= 0.0 or outs[i - 1] != 1:
            continue
        pts.append((gap - air, outs[i], budgets[i]))
    if len(pts) < 20:
        print("success by rest time: too few pairs with a logged gap_s (need >= 20)")
        return
    base = [(b, o) for r, o, b in pts if r >= 0.6 and b > 0]
    if len(base) >= 30:
        bs = fit_sigma([b for b, _ in base], [o for _, o in base])[0]
        note = f"model sigma {bs} ms, fitted on {len(base)} pairs with rest >= 0.6 s"
    else:
        bs = sigma_ms
        note = f"model sigma {bs} ms, fitted on all obstacles (few long-rest pairs)"
    print("success by rest time (previous obstacle cleared; rest = gap_s - air time;"
          " rest < 0 needs perfect timing):")
    print(f"  {note}")
    print(f"  {'rest (s)':>9s} {'n':>4s} {'obs':>6s} {'95% CI':>14s} {'model':>6s}")
    below = []
    for lo_, hi_ in zip(REST_EDGES[:-1], REST_EDGES[1:]):
        sel = [(o, b) for r, o, b in pts if lo_ <= r < hi_]
        if not sel:
            continue
        kk = sum(o for o, _ in sel)
        l, h = wilson(kk, len(sel))
        pred = st.mean(p_clear(b, bs / 1000.0) for _, b in sel)
        flag = ""
        if len(sel) < MIN_BIN_N:
            flag = " (few)"
        elif wilson(kk, len(sel), z=2.8)[1] < pred:  # stricter: ~8 bins are compared
            flag = "  << below model"
            below.append(_rest_label(lo_, hi_))
        print(f"  {_rest_label(lo_, hi_):>9s} {len(sel):4d} {kk / len(sel):6.2f} "
              f"[{l:.2f},{h:.2f}] {pred:6.2f}{flag}")
    if below:
        print("  bins clearly below the timing-only model (99.5% bound): " + ", ".join(below)
              + "  -> recovery/decision time limits these gaps")
    else:
        print("  no bin clearly below the timing-only model in this range")


def outcome_group(label, rows, a, meta=None):
    """Summarise one player/session group. Returns a one-line table row or None."""
    i0 = a.trim_start
    i1 = len(rows) - a.trim_end
    kept = rows[i0:i1] if i1 > i0 else []
    print(f"\n--- {label} ---")
    if meta:
        print(_meta_line(meta))
    print(f"rows: {len(rows)} total, {len(kept)} kept "
          f"(trim {a.trim_start} at start, {a.trim_end} at end)")
    n = len(kept)
    if n < 5:
        print("too few obstacles after trimming")
        return None

    outs = [int(float(r["outcome"])) for r in kept]
    k = sum(outs)
    lo, hi = wilson(k, n)
    verdict = ("target inside interval" if lo <= a.target <= hi
               else "ABOVE target (too easy?)" if lo > a.target
               else "BELOW target (too hard?)")
    print(f"obstacles {n}: cleared {k}, crashed {n - k}")
    print(f"success {k / n:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  target {a.target:.2f}: {verdict}")

    half = n // 2
    if half >= 5:
        print(f"first half {sum(outs[:half]) / half:.2f}   second half "
              f"{sum(outs[half:]) / (n - half):.2f}   (large gap = learning or fatigue)")

    d_tail_mean = None
    if "d_spawn" in kept[0]:
        d = [float(r["d_spawn"]) for r in kept]
        t = min(TAIL, n)
        tail = d[-t:]
        tm = st.mean(tail)
        d_tail_mean = tm
        settled = next((i for i in range(n) if all(abs(x - tm) <= BAND for x in d[i:])), None)
        at_max = sum(x >= 0.99 for x in tail) / t
        at_min = sum(x <= 0.01 for x in tail) / t
        print(f"d: mean {st.mean(d):.3f}  min {min(d):.3f}  max {max(d):.3f}")
        print(f"d last {t}: mean {tm:.3f}  std {st.stdev(tail) if t > 1 else 0:.3f}  "
              f"settled(+-{BAND}) from obstacle "
              f"{('#' + str(settled)) if settled is not None else 'never'}")
        print(f"d pinned in tail: at max {at_max:.0%}, at min {at_min:.0%}")

    # model fit
    sigma_line = ""
    budgets = []
    try:
        budgets = [budget_s(r) for r in kept]
    except (KeyError, ValueError):
        budgets = []
    if budgets and any(b > 0 for b in budgets):
        mb = st.mean(b for b in budgets if b > 0) * 1000.0
        sg, slo, shi, glo, ghi = fit_sigma(budgets, outs)
        print(f"timing budget B: mean {mb:.0f} ms (from logged speed/height/width)")
        if sg <= glo + 1 or slo <= glo + 1:
            print(f"fitted sigma: at most {shi} ms (nearly all cleared; only an upper bound is "
                  "identifiable, the real value may be much smaller)")
            sigma_line = f"<= {shi} (bound)"
        elif sg >= ghi - 1 or shi >= ghi - 1:
            print(f"fitted sigma: at least {slo} ms (nearly all crashed; only a lower bound is "
                  "identifiable)")
            sigma_line = f">= {slo} (bound)"
        else:
            print(f"fitted sigma: {sg} ms   95% interval [{slo}, {shi}] ms")
            sigma_line = f"{sg} [{slo},{shi}]"

        # observed vs predicted by d bin (model check)
        if "d_spawn" in kept[0]:
            d = [float(r["d_spawn"]) for r in kept]
            edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
            print("success by d (observed vs model prediction at fitted sigma):")
            print(f"  {'d range':>9s} {'n':>4s} {'obs':>6s} {'95% CI':>14s} {'model':>6s}")
            for a_, b_ in zip(edges[:-1], edges[1:]):
                idx = [i for i, x in enumerate(d) if a_ <= x < b_]
                if not idx:
                    continue
                kk = sum(outs[i] for i in idx)
                l, h = wilson(kk, len(idx))
                pred = st.mean(p_clear(budgets[i], sg / 1000.0) for i in idx)
                few = "" if len(idx) >= MIN_BIN_N else " (few)"
                print(f"  {a_:3.1f}-{min(b_, 1.0):<4.1f} {len(idx):4d} {kk / len(idx):6.2f} "
                      f"[{l:.2f},{h:.2f}] {pred:6.2f}{few}")
        if "gap_s" in kept[0]:
            rest_table(kept, outs, budgets, sg)
    return (label, n, k / n, lo, hi, d_tail_mean, sigma_line)


# ---------------------------------------------------------- latency report
def latency_group(label, rows, a, meta=None):
    kept = rows[a.trim_start: len(rows) - a.trim_end] if len(rows) - a.trim_end > a.trim_start else []
    print(f"\n--- {label} [latency] ---")
    if meta:
        print(_meta_line(meta))
    print(f"rows: {len(rows)} total, {len(kept)} kept")
    if len(kept) < 5:
        print("too few rows after trimming")
        return
    print(f"  {'metric':24s}{'mean':>8s}{'median':>8s}{'p90':>8s}{'max':>8s}{'std':>8s}")
    for c in LAT_COLS:
        v = [float(r[c]) for r in kept]
        print(f"  {c:24s}{st.mean(v):8.2f}{st.median(v):8.2f}{percentile(v, 0.9):8.2f}"
              f"{max(v):8.2f}{(st.stdev(v) if len(v) > 1 else 0):8.2f}")
    acc = [int(float(r["accepted"])) for r in kept]
    loop_med = st.median(float(r["loop_ms"]) for r in kept)
    print(f"jumps accepted {sum(acc)}, rejected (airborne) {len(acc) - sum(acc)}; "
          f"loop {1000 / loop_med:.1f} FPS (median {loop_med:.1f} ms)")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="CSV logs (outcome and/or latency)")
    ap.add_argument("--trim-start", type=int, default=10, metavar="N",
                    help="drop first N rows of each player/session group (default 10)")
    ap.add_argument("--trim-end", type=int, default=0, metavar="N",
                    help="drop last N rows of each group (default 0)")
    ap.add_argument("--target", type=_target_rate, default=0.75, help="target success rate")
    ap.add_argument("--anon", action="store_true", help="replace player names with P1, P2, ...")
    ap.add_argument("--out", metavar="FILE", help="also write the summary to this text file")
    a = ap.parse_args()

    lines = []

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            lines.append(s)

        def flush(self):
            sys.__stdout__.flush()

    sys.stdout = Tee()

    print("=== GESTURE LAB SUMMARY v1 ===")
    print(f"repo commit: {git_hash()}   files: {len(a.paths)}   "
          f"trim: start {a.trim_start}, end {a.trim_end}   target: {a.target}")

    table = []
    anon_map = {}

    def who(name):
        if not a.anon:
            return name
        return anon_map.setdefault(name, f"P{len(anon_map) + 1}")

    for p in a.paths:
        meta, rows = load(p)
        kind = kind_of(rows)
        if kind is None:
            print(f"\n{p}: unrecognised columns, skipping")
            continue
        stem = Path(p).stem
        player = who(rows[0].get("player") or stem)
        if kind == "latency":
            latency_group(f"{player} / {stem}", rows, a, meta=meta)
            continue
        if "session" in rows[0]:
            groups = {}
            for r in rows:
                groups.setdefault(r["session"], []).append(r)
            if len(groups) > 1:
                print(f"\nnote: {stem} has {len(groups)} sessions (preset change or restart); "
                      "each is summarised separately")
            for sid in sorted(groups, key=lambda x: int(float(x))):
                res = outcome_group(f"{player} / {stem} / session {sid}", groups[sid], a, meta=meta)
                if res:
                    table.append(res)
        else:
            res = outcome_group(f"{player} / {stem}", rows, a, meta=meta)
            if res:
                table.append(res)

    if table:
        print("\n=== TABLE (one line per player/session) ===")
        print(f"{'group':34s}{'n':>5s}{'succ':>7s}{'95% CI':>15s}{'d_tail':>8s}  sigma ms [95%]")
        for label, n, s, lo, hi, dt, sig in table:
            dts = f"{dt:.2f}" if dt is not None else "-"
            print(f"{label[:33]:34s}{n:5d}{s:7.2f}  [{lo:.2f},{hi:.2f}]{dts:>8s}  {sig or '-'}")
        print("(\"bound\" = nearly all cleared or all crashed: sigma is only bounded, not estimated)")
    print("\n=== END SUMMARY ===")

    sys.stdout = sys.__stdout__
    if a.out:
        Path(a.out).write_text("".join(lines))
        print(f"\n(summary also written to {a.out})")


if __name__ == "__main__":
    main()