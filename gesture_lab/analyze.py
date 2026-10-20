#!/usr/bin/env python3
"""Analyze logs from the gesture runner game.

Two CSV types are detected automatically from the header:

  latency  : frame_read_to_jump_ms, first_open_to_jump_ms, inference_ms,
             loop_ms, accepted           (written by --log-latency)
  outcomes : t_s, player, session, mode, d_spawn, d_after, speed, height,
             width, outcome (1 = cleared, 0 = crashed)   (written by --log-outcomes)

Examples:
    python -m gesture_lab.analyze run1.csv
    python -m gesture_lab.analyze run1.csv --trim-start 5 --trim-end 5 --plot report.png
    python -m gesture_lab.analyze data/*.csv --trim-start-s 20 --trim-end-s 10

Trimming: --trim-start/--trim-end drop N rows; --trim-start-s/--trim-end-s drop
seconds (needs a t_s column). Trimmed rows are shaded grey in the plots.

Everything here describes the logged sessions only. Nothing is calibrated until
enough different players have been logged.
"""
import argparse
import csv
import math
import statistics as st
import sys
from pathlib import Path

LAT_COLS = ["frame_read_to_jump_ms", "first_open_to_jump_ms", "inference_ms", "loop_ms"]


def _target_rate(value):
    v = float(value)
    if not 0.0 < v < 1.0:
        raise argparse.ArgumentTypeError(f"target success rate must be in (0, 1), got {v}")
    return v


BAND = 0.15        # "settled" means d stays within +-BAND of its late-session mean
MIN_BIN_N = 10     # bins with fewer obstacles are shown but not used for conclusions


# ----------------------------------------------------------------- helpers
def percentile(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = q * (len(s) - 1)
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] * (c - k) + s[c] * (k - f)


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return lo, hi


def describe(v):
    if not v:
        return dict(n=0)
    return dict(n=len(v), mean=st.mean(v), med=st.median(v), p90=percentile(v, 0.9),
                p95=percentile(v, 0.95), mx=max(v), sd=st.stdev(v) if len(v) > 1 else 0.0)


def robust_sd(v):
    m = st.median(v)
    return 1.4826 * st.median([abs(x - m) for x in v]) or 1e-9


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


def trim_range(rows, a):
    """Return (i0, i1): the kept slice rows[i0:i1]."""
    n = len(rows)
    i0, i1 = 0, n
    if "t_s" in rows[0]:
        ts = [float(r["t_s"]) for r in rows]
        if a.trim_start_s:
            i0 = next((i for i, t in enumerate(ts) if t >= a.trim_start_s), n)
        if a.trim_end_s:
            cutoff = ts[-1] - a.trim_end_s
            i1 = max((i + 1 for i, t in enumerate(ts) if t <= cutoff), default=0)
    elif a.trim_start_s or a.trim_end_s:
        print("  note: no t_s column in this file, ignoring --trim-*-s (use --trim-start/--trim-end)")
    i0 += a.trim_start
    i1 = min(i1, n) - a.trim_end
    return min(i0, n), max(min(i0, n), i1)


def col(rows, name):
    return [float(r[name]) for r in rows]


# ----------------------------------------------------------------- latency
def design_margins():
    """Timing budget per difficulty, from the game's own physics (adaptive.py)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from gesture_lab.adaptive import Config, difficulty_profile, timing_window
    except Exception:
        return None
    cfg = Config()
    out = []
    for d in (0.0, 0.5, 1.0):
        p = difficulty_profile(d, cfg)
        h = p.height * (cfg.HEIGHT_JITTER_MEAN + 2.0 * cfg.HEIGHT_JITTER_STD)  # +2σ jittered obstacle
        window = timing_window(h, p.width, p.speed, cfg) * 1000.0
        out.append((d, window, 0.5 * window))         # clears_obstacle allows +-half
    return out


def latency_report(rows, i0, i1, name, plot):
    full_fo = col(rows, "first_open_to_jump_ms")
    kept = rows[i0:i1]
    print(f"\n=== {name}  [latency log]  rows: {len(rows)} total, {len(kept)} kept "
          f"(dropped {i0} at start, {len(rows) - i1} at end)")
    if len(kept) < 5:
        print("  too few rows after trimming")
        return

    print(f"\n  {'metric':24s}{'n':>5s}{'mean':>9s}{'median':>9s}{'p90':>9s}{'p95':>9s}{'max':>9s}{'std':>9s}")
    stats = {}
    for c in LAT_COLS:
        d = describe(col(kept, c))
        stats[c] = d
        print(f"  {c:24s}{d['n']:5d}{d['mean']:9.2f}{d['med']:9.2f}{d['p90']:9.2f}"
              f"{d['p95']:9.2f}{d['mx']:9.2f}{d['sd']:9.2f}")

    acc = [int(float(r["accepted"])) for r in kept]
    print(f"\n  jumps accepted: {sum(acc)}   rejected (airborne): {len(acc) - sum(acc)}")
    loop_med = stats["loop_ms"]["med"]
    fo = stats["first_open_to_jump_ms"]
    print(f"  effective loop rate: {1000 / loop_med:.1f} FPS (median loop {loop_med:.1f} ms)")
    ratio = st.median([a / b for a, b in zip(col(kept, "first_open_to_jump_ms"), col(kept, "loop_ms")) if b > 0])
    print(f"  first_open->jump = {ratio:.2f} loop periods  (about stability_frames - 1)")

    # outliers and edge check
    v = col(kept, "first_open_to_jump_ms")
    m, s = st.median(v), robust_sd(v)
    out = [(i + i0, round(x, 1)) for i, x in enumerate(v) if abs(x - m) > 3.5 * s]
    print(f"  outliers in first_open_to_jump (>3.5 robust sd): {len(out)}"
          + (f"  e.g. rows {out[:5]}" if out else ""))
    k = 10
    if len(full_fo) >= 3 * k:
        mid = full_fo[k:-k]
        mm, ms = st.median(mid), robust_sd(mid)
        print(f"  edge check on UNtrimmed data: first {k} rows are {(st.median(full_fo[:k]) - mm) / ms:+.1f}, "
              f"last {k} rows are {(st.median(full_fo[-k:]) - mm) / ms:+.1f} robust sd from the middle "
              f"(|x| > 3 suggests the edges differ)")

    margins = design_margins()
    if margins:
        jit_sd, jit_tail = fo["sd"], fo["p90"] - fo["med"]
        print("\n  Software timing jitter vs. the game's timing budget (+-half of net window):")
        print(f"  {'d':>5s}{'window ms':>11s}{'budget +-ms':>13s}{'std %':>8s}{'p90-med %':>11s}")
        for d, window, budget in margins:
            print(f"  {d:5.2f}{window:11.0f}{budget:13.0f}{100 * jit_sd / budget:8.1f}{100 * jit_tail / budget:11.1f}")
    print("\n  Scope: this measures software only (frame read -> jump). Camera exposure/buffering,"
          "\n  display delay and human timing noise are NOT in this file, so real latency and"
          "\n  jitter are larger than shown.")

    if plot:
        try:
            _plot_latency(rows, i0, i1, name, plot)
        except Exception as exc:
            print(f"  note: plot could not be drawn: {exc}")


def _plot_latency(rows, i0, i1, name, out):
    plt = _mpl()
    if not plt:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for c in ("first_open_to_jump_ms", "loop_ms", "inference_ms"):
        ax[0].plot(col(rows, c), label=c, lw=1)
    ax[0].axvspan(-0.5, i0 - 0.5, color="grey", alpha=0.3)
    ax[0].axvspan(i1 - 0.5, len(rows) - 0.5, color="grey", alpha=0.3)
    ax[0].set(title="per jump (grey = trimmed)", xlabel="jump #", ylabel="ms")
    ax[0].legend(fontsize=7)
    ax[1].hist(col(rows[i0:i1], "first_open_to_jump_ms"), bins=20)
    ax[1].set(title="first_open -> jump (kept rows)", xlabel="ms")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"  plot saved: {out}")


# ---------------------------------------------------------------- outcomes
def dcol(rows):
    for c in ("d_spawn", "d_after", "d"):
        if c in rows[0]:
            return c
    return None


def _session_plot_path(base, sid):
    if base is None:
        return None
    pp = Path(base)
    return str(pp.with_name(f"{pp.stem}_s{sid}{pp.suffix}"))


def outcomes_report(rows, i0, i1, name, a, plot=None, pooled=False):
    kept = rows[i0:i1]
    n = len(kept)
    total = len(rows)
    if n == 0:
        print(f"\n=== {name}  [outcome log]  rows: {total} total, 0 kept "
              f"(dropped {i0} at start, {len(rows) - i1} at end)")
        print("  too few obstacles after trimming")
        return
    if "session" in kept[0]:
        sessions = {}
        for r in kept:
            sessions.setdefault(int(r["session"]), []).append(r)
        if len(sessions) > 1:
            for sid in sorted(sessions):
                session_plot = _session_plot_path(plot, sid)
                _outcomes_body(sessions[sid], f"{name} session {sid}", a, plot=session_plot,
                               pooled=False, total=total,
                               dropped_start=i0, dropped_end=len(rows) - i1)
            _outcomes_body(kept, f"{name} all sessions", a, plot=None, pooled=True,
                           total=total, dropped_start=i0, dropped_end=len(rows) - i1)
            return
    _outcomes_body(kept, name, a, plot=plot, pooled=pooled,
                   total=total, dropped_start=i0, dropped_end=len(rows) - i1)


def _outcomes_body(kept, name, a, plot=None, pooled=False, total=None, dropped_start=0, dropped_end=0):
    tag = "pooled" if pooled else "outcome log"
    total = total if total is not None else len(kept)
    print(f"\n=== {name}  [{tag}]  rows: {total} total, {len(kept)} kept "
          f"(dropped {dropped_start} at start, {dropped_end} at end)")
    n = len(kept)
    if n < 5:
        print("  too few obstacles after trimming")
        return
    out = [int(float(r["outcome"])) for r in kept]
    k = sum(out)
    lo, hi = wilson(k, n)
    verdict = ("target inside the interval" if lo <= a.target <= hi
               else "ABOVE target (too easy?)" if lo > a.target else "BELOW target (too hard?)")
    print(f"\n  obstacles: {n}   cleared: {k}   crashed: {n - k}")
    print(f"  success rate: {k / n:.3f}   95% CI [{lo:.3f}, {hi:.3f}]   target {a.target:.2f}: {verdict}")

    dc = dcol(kept)
    if dc is None:
        print("  no d column; skipping difficulty analysis")
        return
    d = col(kept, dc)
    w = min(a.window, n)
    roll = [sum(out[i:i + w]) / w for i in range(n - w + 1)]
    if not pooled:
        t = min(a.tail, n)
        tail = d[-t:]
        tm = st.mean(tail)
        print(f"\n  difficulty d ({dc}): mean {st.mean(d):.3f}, min {min(d):.3f}, max {max(d):.3f}")
        print(f"  last {t} obstacles: mean {tm:.3f}, std {st.stdev(tail) if t > 1 else 0:.3f}")
        settled = next((i for i in range(n) if all(abs(x - tm) <= BAND for x in d[i:])), None)
        print("  settled (stays within +-%.2f of that mean): %s"
              % (BAND, f"from obstacle #{settled}" if settled is not None else "never"))
        at_max = sum(x >= 0.99 for x in tail) / t
        at_min = sum(x <= 0.01 for x in tail) / t
        if at_max > 0.5:
            print(f"  WARNING: d pinned at the maximum for {at_max:.0%} of the tail: this player is "
                  "better than the hardest setting, so the ceiling is too low (or the player is exceptional).")
        if at_min > 0.5:
            print(f"  WARNING: d pinned at the minimum for {at_min:.0%} of the tail: still too hard at "
                  "the easiest setting.")

        print(f"  rolling success (window {w}): min {min(roll):.2f}, max {max(roll):.2f}, last {roll[-1]:.2f}")

    # p(d) by bin
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    print(f"\n  success by difficulty (observational: d is steered by outcomes in adaptive mode)")
    print(f"  {'d range':>11s}{'n':>6s}{'rate':>8s}{'95% CI':>18s}")
    bins = []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        idx = [i for i, x in enumerate(d) if a_ <= x < b_]
        if not idx:
            continue
        kk = sum(out[i] for i in idx)
        l, h = wilson(kk, len(idx))
        mid = st.mean(d[i] for i in idx)
        bins.append((mid, kk / len(idx), l, h, len(idx)))
        flag = "" if len(idx) >= MIN_BIN_N else "  (few)"
        print(f"  {a_:4.1f}-{min(b_, 1.0):<5.1f}{len(idx):6d}{kk / len(idx):8.2f}   [{l:.2f}, {h:.2f}]{flag}")
    good = [b for b in bins if b[4] >= MIN_BIN_N]
    if len(good) >= 2:
        first, last = good[0], good[-1]
        if last[3] < first[2]:
            print("  trend: success falls as d rises (lowest vs highest bin CIs do not overlap)")
        elif last[2] > first[3]:
            print("  trend: success RISES with d, which breaks the controller's assumption")
        else:
            print("  trend: not distinguishable from flat with this much data")
        cross = None
        for b1, b2 in zip(good[:-1], good[1:]):
            if (b1[1] - a.target) * (b2[1] - a.target) <= 0 and b1[1] != b2[1]:
                cross = b1[0] + (a.target - b1[1]) * (b2[0] - b1[0]) / (b2[1] - b1[1])
                break
        print(f"  p(d) crosses {a.target:.2f} near d = {cross:.2f}" if cross is not None else
              f"  p(d) does not cross {a.target:.2f} within the observed d range "
              f"[{good[0][0]:.2f}, {good[-1][0]:.2f}]")
    else:
        print("  not enough data per bin for a p(d) conclusion")

    if plot and not pooled:
        try:
            _plot_outcomes(d, out, roll, w, dropped_start, bins, name, plot, a.target)
        except Exception as exc:
            print(f"  note: plot could not be drawn: {exc}")
def _plot_outcomes(d, out, roll, w, i0, bins, name, path, target):
    plt = _mpl()
    if not plt:
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(d)
    ax[0].set(title="difficulty d per obstacle", xlabel="obstacle # (kept)", ylim=(0, 1))
    ax[1].plot(range(w - 1, w - 1 + len(roll)), roll)
    ax[1].axhline(target, ls="--", c="r", lw=1)
    ax[1].set(title=f"rolling success (window {w})", xlabel="obstacle # (kept)", ylim=(0, 1))
    if bins:
        ax[2].errorbar([b[0] for b in bins], [b[1] for b in bins],
                       yerr=[[b[1] - b[2] for b in bins], [b[3] - b[1] for b in bins]], fmt="o-", capsize=3)
    ax[2].axhline(target, ls="--", c="r", lw=1)
    ax[2].set(title="success vs d (95% CI)", xlabel="d", xlim=(0, 1), ylim=(0, 1))
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  plot saved: {path}")


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  (matplotlib not installed, skipping plot)")
        return None


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="one or more CSV logs")
    ap.add_argument("--trim-start", type=int, default=0, metavar="N", help="drop first N rows")
    ap.add_argument("--trim-end", type=int, default=0, metavar="N", help="drop last N rows")
    ap.add_argument("--trim-start-s", type=float, default=0.0, metavar="S", help="drop rows before t_s = S")
    ap.add_argument("--trim-end-s", type=float, default=0.0, metavar="S", help="drop the last S seconds")
    ap.add_argument("--target", type=_target_rate, default=0.75, help="target success rate (default 0.75)")
    ap.add_argument("--tail", type=int, default=30, help="obstacles used for the late-session d (default 30)")
    ap.add_argument("--window", type=int, default=20, help="rolling success window (default 20)")
    ap.add_argument("--plot", metavar="PNG", help="save plots; one file per input: <stem>_<input>.png")
    a = ap.parse_args()

    pooled_rows = []
    for p in a.paths:
        meta, rows = load(p)
        kind = kind_of(rows)
        if meta:
            print(f"\n{Path(p).name} metadata:")
            for k in ("ADAPTATION_RATE", "MOMENTUM", "TARGET_SUCCESS_RATE", "D_START", "effective_start_d",
                      "initial_preset", "player", "timestamp", "git_hash"):
                if k in meta:
                    print(f"  {k} = {meta[k]}")
        if kind is None:
            print(f"\n{p}: unrecognised columns, skipping")
            continue
        i0, i1 = trim_range(rows, a)
        plot = None
        if a.plot:
            pp = Path(a.plot)
            plot = str(pp.with_name(f"{pp.stem}_{Path(p).stem}{pp.suffix or '.png'}")) if len(a.paths) > 1 else a.plot
        if kind == "latency":
            latency_report(rows, i0, i1, Path(p).name, plot)
        else:
            outcomes_report(rows, i0, i1, Path(p).name, a, plot)
            pooled_rows += rows[i0:i1]
    if len(a.paths) > 1 and pooled_rows:
        outcomes_report(pooled_rows, 0, len(pooled_rows), "all outcome files", a, pooled=True)


if __name__ == "__main__":
    main()