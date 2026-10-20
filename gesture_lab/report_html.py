#!/usr/bin/env python3
"""One self-contained HTML page about a player's jump timing.

    python report_html.py data/alice_run1.csv --out report.html
    python -m gesture_lab.analyze_trials data/alice_run1.csv --html report.html   # same thing

The text report (``analyze_trials.py``) stays the place for quick numbers.  This
page adds pictures and one extra model that only lives here: **drift**.

Drift = does the player's bias change during the session (warm-up, tiredness)?
We use a Kalman-style "local level" model:

    bias_t  = bias_(t-1) + w      w ~ N(0, q^2)     (the bias may wander a little)
    delta_t = bias_t + e          e ~ N(0, r^2)     (each jump is the bias plus noise)

q and r are chosen by maximum likelihood on a small grid.  q = 0 means "no drift".
Plain Python, no extra packages, so every step can be read.
"""
from __future__ import annotations

import argparse
import html
import math
import statistics
import sys
from typing import Optional, Sequence

from gesture_lab.analyze_trials import _flag, _num, load_trials

# ---------------------------------------------------------------------------
# Small maths helpers (milliseconds everywhere in this file)
# ---------------------------------------------------------------------------


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def p_clear(window_ms: float, mu: float, sigma: float) -> float:
    """Chance a Gaussian jump (bias mu, spread sigma) lands inside a window."""
    h = window_ms / 2.0
    return _phi((h - mu) / sigma) - _phi((-h - mu) / sigma)


def window_for_rate(rate: float, mu: float, sigma: float) -> float:
    """Smallest window (ms) that gives ``rate`` success, by bisection."""
    lo, hi = 0.0, 2000.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if p_clear(mid, mu, sigma) < rate else (lo, mid)
    return hi


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------------------
# Drift: Kalman filter + smoother for a slowly moving bias
# ---------------------------------------------------------------------------


def _kalman(y: Sequence[float], q: float, r: float):
    """Filter y (ms). Returns (loglik, filtered m, P, predicted m, P)."""
    m, p = 0.0, 100.0 ** 2           # diffuse start
    ll = 0.0
    mf, pf, mp, pp = [], [], [], []
    for i, obs in enumerate(y):
        m_pred, p_pred = m, p + (q * q if i else 0.0)
        f = p_pred + r * r
        v = obs - m_pred
        ll += -0.5 * (math.log(2 * math.pi * f) + v * v / f)
        k = p_pred / f
        m, p = m_pred + k * v, (1 - k) * p_pred
        mf.append(m); pf.append(p); mp.append(m_pred); pp.append(p_pred)
    return ll, mf, pf, mp, pp


def fit_drift(y: Sequence[float]) -> dict:
    """Pick q (drift per jump) and r (jump noise) by likelihood; smooth the path."""
    q_grid = [0, 1, 2, 3, 5, 8, 12, 20]
    r_grid = [x for x in range(15, 160, 5)]
    best = {}
    for q in q_grid:
        ll, r = max((_kalman(y, q, r)[0], r) for r in r_grid)
        best[q] = (ll, r)
    q_hat = max(best, key=lambda q: best[q][0])
    ll_hat, r_hat = best[q_hat]
    _, mf, pf, mp, pp = _kalman(y, q_hat, r_hat)
    ms, ps = mf[:], pf[:]            # Rauch-Tung-Striebel smoother
    for t in range(len(y) - 2, -1, -1):
        j = pf[t] / pp[t + 1] if pp[t + 1] > 0 else 0.0
        ms[t] = mf[t] + j * (ms[t + 1] - mp[t + 1])
        ps[t] = pf[t] + j * j * (ps[t + 1] - pp[t + 1])
    gain = 2.0 * (ll_hat - best[0][0])       # likelihood-ratio statistic vs "no drift"
    return {"q": q_hat, "r": r_hat, "mean": ms, "sd": [math.sqrt(max(v, 0)) for v in ps],
            "lr": gain, "drifting": gain > 4.0}


# ---------------------------------------------------------------------------
# Tiny SVG helpers
# ---------------------------------------------------------------------------

W, H, ML, MR, MT, MB = 720, 300, 52, 14, 16, 36


class Axes:
    def __init__(self, x0, x1, y0, y1):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.parts: list[str] = []

    def X(self, x):
        return ML + (x - self.x0) / (self.x1 - self.x0 or 1) * (W - ML - MR)

    def Y(self, y):
        return H - MB - (y - self.y0) / (self.y1 - self.y0 or 1) * (H - MT - MB)

    def add(self, s):
        self.parts.append(s)

    def grid(self, yticks, xticks, xlabel, ylabel):
        for v in yticks:
            self.add(f'<line class="g" x1="{ML}" x2="{W - MR}" y1="{self.Y(v):.1f}" y2="{self.Y(v):.1f}"/>'
                     f'<text class="t" x="{ML - 6}" y="{self.Y(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
        for v in xticks:
            self.add(f'<text class="t" x="{self.X(v):.1f}" y="{H - MB + 16}" text-anchor="middle">{v:g}</text>')
        self.add(f'<text class="t" x="{(ML + W - MR) / 2}" y="{H - 4}" text-anchor="middle">{html.escape(xlabel)}</text>')
        self.add(f'<text class="t" transform="rotate(-90 12 {H / 2})" x="12" y="{H / 2}" text-anchor="middle">{html.escape(ylabel)}</text>')

    def svg(self):
        return f'<svg viewBox="0 0 {W} {H}" role="img">' + "".join(self.parts) + "</svg>"


def _ticks(lo, hi, n=5):
    step = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(step)) if step > 0 else 1
    step = min((s * mag for s in (1, 2, 5, 10) if s * mag >= step), default=step)
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + 1e-9:
        out.append(round(v, 6)); v += step
    return out


def chart_timeline(jumps, drift) -> str:
    """Every jump against its own safe window, plus the drifting bias."""
    lim = max(150.0, max(abs(j["delta"]) for j in jumps) * 1.1,
              max(j["W"] / 2 for j in jumps) * 1.05)
    a = Axes(1, max(len(jumps), 2), -lim, lim)
    a.grid(_ticks(-lim, lim), [x for x in _ticks(1, len(jumps), 6) if x >= 1], "jump number", "ms from window centre (early < 0 < late)")
    a.add(f'<line class="z" x1="{ML}" x2="{W - MR}" y1="{a.Y(0):.1f}" y2="{a.Y(0):.1f}"/>')
    for i, j in enumerate(jumps, 1):         # safe window of each obstacle
        a.add(f'<line class="win" x1="{a.X(i):.1f}" x2="{a.X(i):.1f}" y1="{a.Y(-j["W"] / 2):.1f}" y2="{a.Y(j["W"] / 2):.1f}"/>')
    if drift:                                 # drift band and line
        up = [(a.X(i), a.Y(m + s)) for i, (m, s) in enumerate(zip(drift["mean"], drift["sd"]), 1)]
        dn = [(a.X(i), a.Y(m - s)) for i, (m, s) in enumerate(zip(drift["mean"], drift["sd"]), 1)]
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in up + dn[::-1])
        a.add(f'<polygon class="band" points="{pts}"/>')
        line = " ".join(f"{a.X(i):.1f},{a.Y(m):.1f}" for i, m in enumerate(drift["mean"], 1))
        a.add(f'<polyline class="trend" points="{line}"/>')
    for i, j in enumerate(jumps, 1):
        cls = "ok" if j["ok"] else "bad"
        a.add(f'<circle class="{cls}" cx="{a.X(i):.1f}" cy="{a.Y(j["delta"]):.1f}" r="4.5"><title>jump {i}: {j["delta"]:.0f} ms, window {j["W"]:.0f} ms</title></circle>')
    return a.svg()


def chart_success(rows, mu, sigma) -> str:
    """Observed success by window width (Wilson ranges) against the model curve."""
    rows = sorted(rows, key=lambda r: r["W"])
    k = max(3, min(7, len(rows) // 4))
    groups = [rows[i * len(rows) // k:(i + 1) * len(rows) // k] for i in range(k)]
    groups = [g for g in groups if g]
    xmax = max(r["W"] for r in rows) * 1.05
    a = Axes(0, xmax, 0, 1)
    a.grid([0, .25, .5, .75, 1], _ticks(0, xmax, 5), "safe window width (ms)", "chance of clearing")
    if sigma and sigma == sigma:
        pts = " ".join(f"{a.X(w):.1f},{a.Y(p_clear(w, mu, sigma)):.1f}" for w in [xmax * i / 60 for i in range(61)])
        a.add(f'<polyline class="trend" points="{pts}"/>')
    for g in groups:
        n, kk = len(g), sum(r["ok"] for r in g)
        w = statistics.fmean(r["W"] for r in g)
        lo, hi = wilson(kk, n)
        a.add(f'<line class="win" x1="{a.X(w):.1f}" x2="{a.X(w):.1f}" y1="{a.Y(lo):.1f}" y2="{a.Y(hi):.1f}"/>'
              f'<circle class="ok" cx="{a.X(w):.1f}" cy="{a.Y(kk / n):.1f}" r="5"><title>{kk}/{n} cleared, window about {w:.0f} ms</title></circle>')
    return a.svg()


def chart_hist(deltas, mu, sigma) -> str:
    lim = max(200.0, max(abs(d) for d in deltas) * 1.1)
    bins = 12
    width = 2 * lim / bins
    counts = [0] * bins
    for d in deltas:
        counts[min(bins - 1, int((d + lim) / width))] += 1
    a = Axes(-lim, lim, 0, max(counts) + 1)
    a.grid(_ticks(0, max(counts) + 1, 4), _ticks(-lim, lim), "ms from window centre", "jumps")
    for i, c in enumerate(counts):
        x0, x1 = -lim + i * width, -lim + (i + 1) * width
        a.add(f'<rect class="bar" x="{a.X(x0) + 1:.1f}" y="{a.Y(c):.1f}" width="{a.X(x1) - a.X(x0) - 2:.1f}" height="{a.Y(0) - a.Y(c):.1f}"/>')
    if sigma and sigma == sigma:
        pts = " ".join(f"{a.X(x):.1f},{a.Y(len(deltas) * width * math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))):.1f}"
                       for x in [-lim + 2 * lim * i / 80 for i in range(81)])
        a.add(f'<polyline class="trend" points="{pts}"/>')
    return a.svg()


# ---------------------------------------------------------------------------
# Build the page
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6b7280;--card:#f5f5f7;--ok:#2e9e5b;--bad:#d64545;--acc:#3b6fd4;--grid:#e3e3e8}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#ececf0;--mut:#9aa0ab;--card:#222226;--grid:#33343a;--acc:#7da2f0}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:780px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:1.6rem;margin:.2em 0}h2{font-size:1.15rem;margin:1.8em 0 .3em}
.mut{color:var(--mut)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.card{background:var(--card);border-radius:10px;padding:12px 14px}.card b{display:block;font-size:1.5rem}
svg{width:100%;height:auto;background:var(--card);border-radius:10px}
.g{stroke:var(--grid)}.z{stroke:var(--mut);stroke-dasharray:4 3}.t{fill:var(--mut);font-size:11px}
.win{stroke:var(--acc);stroke-opacity:.35;stroke-width:5;stroke-linecap:round}.ok{fill:var(--ok)}.bad{fill:var(--bad)}
.trend{fill:none;stroke:var(--acc);stroke-width:2.2}.band{fill:var(--acc);fill-opacity:.15}.bar{fill:var(--acc);fill-opacity:.55}
table{border-collapse:collapse;width:100%}td,th{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right}th:first-child,td:first-child{text-align:left}
"""


def _jumps(rows):
    out = []
    for r in rows:
        d, w = _num(r, "delta"), _num(r, "W")
        if d is not None and w is not None:
            h = _num(r, "delta_open")
            out.append({"delta": d * 1000, "W": w * 1000, "ok": _flag(r, "outcome"),
                        "speed": _num(r, "speed") or 0.0,
                        "human": (d if h is None else h) * 1000, "has_open": h is not None})
    return out


def build_html(rows: Sequence[dict]) -> str:
    jumps = _jumps(rows)
    feasible = [r for r in rows if _num(r, "W") is not None]
    all_ws = [{"W": _num(r, "W") * 1000, "ok": _flag(r, "outcome")} for r in feasible]
    n_ok = sum(_flag(r, "outcome") for r in rows)
    who = ", ".join(sorted({r.get("player", "") for r in rows if r.get("player")})) or "player"
    parts = [f"<h1>Jump timing report: {html.escape(who)}</h1>",
             f'<p class="mut">{len(rows)} obstacles, {len(jumps)} jumps measured, from {html.escape(", ".join(sorted({r["_source"] for r in rows})))}.</p>']
    if len(jumps) < 8:
        parts.append("<p>Fewer than 8 measured jumps: play a longer session before reading anything into the numbers.</p>")
        return _page("".join(parts))

    deltas = [j["delta"] for j in jumps]          # when the game started each jump: decides the outcome
    mu_g, sigma_g = statistics.fmean(deltas), statistics.stdev(deltas)
    human = [j["human"] for j in jumps]           # when the camera first saw the gesture (software wait removed)
    mu, sigma = statistics.fmean(human), statistics.stdev(human)
    corrected = all(j["has_open"] for j in jumps)
    se_mu, se_sig = sigma / math.sqrt(len(deltas)), sigma / math.sqrt(2 * len(deltas))
    pc = sum(1 for r in feasible if _flag(r, "predicted_success") == _flag(r, "outcome") and _num(r, "t_j") is not None)
    pj = sum(1 for r in feasible if _num(r, "t_j") is not None)
    drift = fit_drift(deltas)

    parts.append('<div class="cards">'
                 f'<div class="card">cleared<b>{n_ok}/{len(rows)}</b>{100 * n_ok / len(rows):.0f}%</div>'
                 f'<div class="card">bias<b>{mu:+.0f} ms</b>±{se_mu:.0f} ({"early" if mu < 0 else "late"})</div>'
                 f'<div class="card">spread<b>{sigma:.0f} ms</b>±{se_sig:.0f}</div>'
                 f'<div class="card">physics check<b>{pc}/{pj}</b>outcomes predicted</div></div>')
    tip = ("Wait a beat before opening your hand." if mu < -15 else
           "Open your hand a little sooner." if mu > 15 else "Your timing is well centred on average.")
    if corrected:
        parts.append("<p class=\"mut\">Bias and spread are your timing from when the camera first saw your gesture "
                     f"(minus half a camera frame), so the game's own wait is removed. The game then started your jumps "
                     f"{mu_g - mu:.0f} ms later on average, and that later time decides clear or crash, so the first chart "
                     "and the success curve use it.</p>")
    parts.append(f"<p>On average your jumps land <b>{abs(mu):.0f} ms {'early' if mu < 0 else 'late'}</b> relative to the middle "
                 f"of the safe window, and about two out of three land within <b>{sigma:.0f} ms</b> of your own average. {tip}</p>")

    parts.append("<h2>Every jump against its safe window</h2>"
                 '<p class="mut">Each pale bar is one obstacle\'s safe window; the dot is where you jumped (green cleared, red crashed). '
                 "The line is the drift estimate described below.</p>" + chart_timeline(jumps, drift))
    verdict = ("The model finds <b>some drift</b>: your bias moved during the session (warming up or getting tired)."
               if drift["drifting"] else
               "The model finds <b>no clear drift</b>: a flat bias explains the session about as well as a moving one.")
    parts.append("<h2>Drift (does your bias change during the session?)</h2>"
                 f"<p>{verdict} Fitted wander: {drift['q']:.0f} ms per jump; jump noise {drift['r']:.0f} ms; "
                 f"likelihood gain over a flat bias {drift['lr']:.1f} (about 4 or more suggests real drift). "
                 "With about 30 jumps the shaded band is wide, so only a large, steady move would show.</p>")
    parts.append("<h2>Chance of clearing by window width</h2>"
                 '<p class="mut">Dots: what happened, with 95% ranges. Line: what a bell curve with your bias and spread predicts.</p>'
                 + chart_success(all_ws, mu_g, sigma_g))
    parts.append("<h2>Where your jumps landed</h2>"
                 '<p class="mut">Bars: your jumps. Line: the bell curve fitted to them.</p>' + chart_hist(human, mu, sigma))

    speeds = sorted({round(j["speed"]) for j in jumps if j["speed"]})
    if len(speeds) > 1:
        body = ""
        for s in speeds:
            sel = [j for j in jumps if round(j["speed"]) == s]
            if len(sel) >= 4:
                hs, gs = [j["human"] for j in sel], [j["delta"] for j in sel]
                m, sd = statistics.fmean(hs), statistics.stdev(hs)
                mg, sg = statistics.fmean(gs), statistics.stdev(gs)        # the window needed uses the game clock
                body += (f"<tr><td>{s} px/s</td><td>{len(sel)}</td><td>{m:+.0f}</td><td>{sd:.0f}</td>"
                         f"<td>{window_for_rate(0.7, mg, sg):.0f}</td></tr>")
        parts.append("<h2>By speed</h2><table><tr><th>speed</th><th>jumps</th><th>bias (ms)</th><th>spread (ms)</th>"
                     "<th>window for 70% (ms)</th></tr>" + body + "</table>"
                     '<p class="mut">The last column is the narrowest safe window you would clear 70% of the time at that speed.</p>')
    parts.append('<p class="mut">Bias includes camera and detection delay, so it describes this player on this setup. '
                 "Numbers after ± are one standard error.</p>")
    return _page("".join(parts))


def _page(body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>Jump timing report</title>'
            f"<style>{CSS}</style></head><body><main>{body}</main></body></html>")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Write a self-contained HTML timing report.")
    ap.add_argument("files", nargs="+", help="trial CSV file(s)")
    ap.add_argument("--out", default="report.html", help="output HTML path")
    args = ap.parse_args(argv)
    rows = load_trials(args.files)
    if not rows:
        print("No trials found.")
        return 1
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build_html(rows))
    print(f"HTML report saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
