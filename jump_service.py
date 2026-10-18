"""Jump-success services: physics window + player timing model.

Pure Python (no pygame, cv2, mediapipe).  Implements the model in
dino_jump_success_model.md:

    P(success | x, h) = F(U) - F(L)

* ``success_window`` is deterministic physics: the set [L, U] of jump times
  (seconds, t = 0 when the obstacle's front edge meets the player's front edge)
  that clear the obstacle and land before the next one.
* A ``TimingModel`` describes the player's timing error
  ``delta = t_j - (L + U) / 2`` and supplies its CDF.
* ``solve_difficulty`` picks the d whose predicted success matches a target.

All geometry and physics come from ``adaptive.Config``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from adaptive import Config, difficulty_profile, jump_peak, air_time


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Kinematics:
    """Crossing times of the obstacle-top height h (seconds after takeoff)."""
    r: float        # |vertical speed| at height h
    t1: float       # rises above h
    t2: float       # drops below h
    t_air: float    # total airtime = t1 + t2
    t_above: float  # t2 - t1 = 2r/g


def kinematics(h: float, cfg: Optional[Config] = None) -> Optional[Kinematics]:
    """Return crossing times for obstacle height h, or None if h >= apex."""
    cfg = cfg or Config()
    v0, g = -cfg.JUMP_VELOCITY, cfg.GRAVITY
    disc = v0 * v0 - 2.0 * g * max(h, 0.0)
    if disc <= 0.0:
        return None
    r = math.sqrt(disc)
    return Kinematics(r=r, t1=(v0 - r) / g, t2=(v0 + r) / g,
                      t_air=2.0 * v0 / g, t_above=2.0 * r / g)


def overlap_time(w: float, speed: float, cfg: Optional[Config] = None) -> float:
    """tau = (w_o + w_p) / s."""
    cfg = cfg or Config()
    return (w + cfg.PLAYER_WIDTH) / speed


def is_clearable(h: float, w: float, speed: float, cfg: Optional[Config] = None) -> bool:
    """h <= v0^2/2g - g tau^2 / 8   (equivalently t_above >= tau)."""
    k = kinematics(h, cfg)
    return k is not None and k.t_above >= overlap_time(w, speed, cfg)


@dataclass(frozen=True)
class JumpWindow:
    L: float
    U_clear: float
    U_gap: float          # +inf when no next obstacle is considered
    U: float
    feasible: bool

    @property
    def width(self) -> float:
        return max(self.U - self.L, 0.0)

    @property
    def centre(self) -> float:
        return 0.5 * (self.L + self.U)


def success_window(
    h: float, w: float, speed: float,
    gap_px: Optional[float] = None, cfg: Optional[Config] = None,
) -> JumpWindow:
    """Hard constraints on the jump time t_j.

    L       = tau - t2
    U_clear = -t1
    U_gap   = (w + gap) / s - T_air      (gap = back of this obstacle to
                                          front of the next one, px)
    U       = min(U_clear, U_gap)
    """
    cfg = cfg or Config()
    k = kinematics(h, cfg)
    if k is None:
        return JumpWindow(math.inf, -math.inf, -math.inf, -math.inf, False)
    tau = overlap_time(w, speed, cfg)
    L = tau - k.t2
    U_clear = -k.t1
    U_gap = math.inf if gap_px is None else (w + gap_px) / speed - k.t_air
    U = min(U_clear, U_gap)
    return JumpWindow(L, U_clear, U_gap, U, U > L)


def height_for_window(
    target_w: float, w: float, speed: float, cfg: Optional[Config] = None,
) -> float:
    """Obstacle height (px) that gives a chosen clearance-window width.

    Start from the clearance-limited window width

        W = 2r/g - tau          (tau = (w + player width) / speed)

    and solve for r, then for the height:

        r = g (W + tau) / 2
        h = (v0^2 - r^2) / (2 g)        because r^2 = v0^2 - 2 g h

    Raises ValueError when no obstacle can give that window (the target is
    wider than a jump over a zero-height obstacle allows, or not positive).
    """
    cfg = cfg or Config()
    v0, g = -cfg.JUMP_VELOCITY, cfg.GRAVITY
    if target_w <= 0.0:
        raise ValueError("target window must be positive")
    tau = overlap_time(w, speed, cfg)
    r = g * (target_w + tau) / 2.0
    if r > v0:
        widest = 2.0 * v0 / g - tau
        raise ValueError(
            f"window {target_w:.3f}s is wider than the maximum {widest:.3f}s "
            f"at speed {speed:g} px/s and width {w:g} px"
        )
    return (v0 * v0 - r * r) / (2.0 * g)


# ---------------------------------------------------------------------------
# Player timing models: distribution of delta = t_j - window centre
# ---------------------------------------------------------------------------

class TimingModel(Protocol):
    def cdf(self, delta: float, speed: float = ...) -> float: ...


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class GaussianTiming:
    """delta ~ N(mu, sigma(s)^2), sigma(s) = sigma0 * (s / s_ref) ** alpha.

    alpha = 0 recovers the plain Gaussian; alpha > 0 makes faster games
    noisier (heteroscedastic upgrade).
    """
    mu: float = 0.0
    sigma0: float = 0.05
    alpha: float = 0.0
    s_ref: float = 400.0

    def sigma(self, speed: float) -> float:
        return max(self.sigma0 * (speed / self.s_ref) ** self.alpha, 1e-6)

    def cdf(self, delta: float, speed: float = 400.0) -> float:
        return _phi((delta - self.mu) / self.sigma(speed))


@dataclass(frozen=True)
class EmpiricalTiming:
    """Smoothed empirical CDF of observed deltas (Gaussian kernel)."""
    deltas: tuple[float, ...]
    bandwidth: float = 0.02

    def cdf(self, delta: float, speed: float = 0.0) -> float:
        if not self.deltas:
            return _phi(delta / max(self.bandwidth, 1e-6))
        b = max(self.bandwidth, 1e-6)
        return sum(_phi((delta - x) / b) for x in self.deltas) / len(self.deltas)


def fit_gaussian(deltas: Sequence[float], min_sigma: float = 0.01) -> GaussianTiming:
    n = len(deltas)
    if n == 0:
        return GaussianTiming()
    mu = sum(deltas) / n
    var = sum((x - mu) ** 2 for x in deltas) / max(n - 1, 1)
    return GaussianTiming(mu=mu, sigma0=max(math.sqrt(var), min_sigma))


def observed_delta(t_j: float, window: JumpWindow) -> float:
    """Timing error of an observed jump, relative to the window centre."""
    return t_j - window.centre


# ---------------------------------------------------------------------------
# Success probability and difficulty solving
# ---------------------------------------------------------------------------

def p_success(window: JumpWindow, model: TimingModel, speed: float) -> float:
    """P(L <= t_j <= U) with t_j = centre + delta."""
    if not window.feasible:
        return 0.0
    half = 0.5 * window.width
    return max(0.0, model.cdf(half, speed) - model.cdf(-half, speed))


def expected_gap_px(speed: float, spawn_interval: float, w: float) -> float:
    """Gap implied by the spawn timer: speed * interval - previous width."""
    return speed * spawn_interval - w


def p_success_at_d(d: float, model: TimingModel, cfg: Optional[Config] = None) -> float:
    """Predicted success probability for a typical obstacle at difficulty d."""
    cfg = cfg or Config()
    prof = difficulty_profile(d, cfg)
    gap = max(cfg.MIN_GAP_FLOOR_PX,
              expected_gap_px(prof.speed, prof.spawn_interval, prof.width))
    win = success_window(prof.height, prof.width, prof.speed, gap, cfg)
    return p_success(win, model, prof.speed)


def solve_difficulty(
    target: float, model: TimingModel, cfg: Optional[Config] = None, steps: int = 200,
) -> float:
    """Largest d in [D_MIN, D_MAX] whose predicted success is >= target.

    P(d) is not guaranteed monotone (speed widens the clearance window but
    narrows the gap window), so scan a grid instead of bisecting.  Returns
    D_MIN if no d reaches the target.
    """
    cfg = cfg or Config()
    best = cfg.D_MIN
    for i in range(steps + 1):
        d = cfg.D_MIN + (cfg.D_MAX - cfg.D_MIN) * i / steps
        if p_success_at_d(d, model, cfg) >= target:
            best = d
    return best
