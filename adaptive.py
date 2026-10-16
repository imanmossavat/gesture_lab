"""Pure-Python adaptive difficulty logic for the gesture runner.

No pygame, cv2, or mediapipe imports.  This module is shared by the game,
the dev bench, and the tests.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional


@dataclass(frozen=True)
class Config:
    """Every tunable number for the adaptive difficulty system.

    The physics constants (GRAVITY, JUMP_VELOCITY) are *measurements* taken from
    the existing game.py.  All other values are *theoretical design choices* and
    must be treated as guesses until they are validated with real players.
    """

    # World / player geometry (pixels)
    PLAYER_X: float = 620.0
    PLAYER_WIDTH: float = 42.0
    PLAYER_HEIGHT: float = 56.0
    GROUND_Y: float = 590.0
    SPAWN_X: float = 1190.0  # WIDTH + 10 from game.py

    # Physics constants -- measured from the existing game file.
    GRAVITY: float = 1800.0
    JUMP_VELOCITY: float = -690.0

    # Difficulty profile -- theoretical mapping from d to game parameters.
    SPEED_MIN: float = 260.0
    SPEED_MAX: float = 430.0  # 260 + 170
    HEIGHT_MIN: float = 30.0
    HEIGHT_MAX: float = 70.0  # 30 + 40
    WIDTH_MIN: float = 20.0
    WIDTH_MAX: float = 35.0  # 20 + 15
    SPAWN_INTERVAL_BASE_AT_D0: float = 2.5
    SPAWN_INTERVAL_BASE_AT_D1: float = 1.3  # 2.5 - 1.2

    # Spawn interval jitter -- multiplicative, only ever lengthens the interval.
    SPAWN_INTERVAL_JITTER: tuple[float, float] = (1.0, 1.3)

    # Height jitter -- theoretical range that adds variation without breaking feasibility.
    # Width is *not* jittered; only height gets ±10% variation.
    HEIGHT_JITTER: float = 0.10

    # Controller -- theoretical choices, not calibrated.
    TARGET_SUCCESS_RATE: float = 0.75
    ADAPTATION_RATE: float = 0.08
    D_MIN: float = 0.0
    D_MAX: float = 1.0
    D_START: float = 0.5
    RECENT_WINDOW: int = 200

    # Minimum gap -- theoretical safety margin.
    LANDING_MARGIN_S: float = 0.35
    MIN_GAP_FLOOR_PX: float = 350.0

    # Latency diagnostics
    LATENCY_WINDOW: int = 200


class DifficultyProfile(NamedTuple):
    speed: float
    height: float
    width: float
    spawn_interval: float


@dataclass
class Obstacle:
    x: float
    w: float
    h: float
    done: bool = False
    failed: bool = False
    d_spawn: float = 0.0
    speed_spawn: float = 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def difficulty_profile(d: float, config: Optional[Config] = None) -> DifficultyProfile:
    """Map normalized difficulty d in [0, 1] to game parameters."""
    cfg = config or Config()
    d = _clamp(d, cfg.D_MIN, cfg.D_MAX)
    speed = cfg.SPEED_MIN + (cfg.SPEED_MAX - cfg.SPEED_MIN) * d
    height = cfg.HEIGHT_MIN + (cfg.HEIGHT_MAX - cfg.HEIGHT_MIN) * d
    width = cfg.WIDTH_MIN + (cfg.WIDTH_MAX - cfg.WIDTH_MIN) * d
    spawn_interval = cfg.SPAWN_INTERVAL_BASE_AT_D0 - (
        cfg.SPAWN_INTERVAL_BASE_AT_D0 - cfg.SPAWN_INTERVAL_BASE_AT_D1
    ) * d
    return DifficultyProfile(speed, height, width, spawn_interval)


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def jump_peak(config: Optional[Config] = None) -> float:
    """Theoretical maximum height of the jump in pixels."""
    cfg = config or Config()
    v = cfg.JUMP_VELOCITY
    return v * v / (2.0 * cfg.GRAVITY)


def air_time(config: Optional[Config] = None) -> float:
    """Theoretical total time the player spends in the air per jump."""
    cfg = config or Config()
    return -2.0 * cfg.JUMP_VELOCITY / cfg.GRAVITY


def apex_time(config: Optional[Config] = None) -> float:
    """Theoretical time from take-off to apex."""
    cfg = config or Config()
    return -cfg.JUMP_VELOCITY / cfg.GRAVITY


def time_above(h: float, config: Optional[Config] = None) -> float:
    """Theoretical time during a jump when the player's feet are >= h px above ground.

    Returns 0.0 for obstacles that are taller than the jump peak.
    """
    cfg = config or Config()
    peak = jump_peak(cfg)
    if h <= 0.0:
        return air_time(cfg)
    if h >= peak:
        return 0.0
    v = cfg.JUMP_VELOCITY
    g = cfg.GRAVITY
    discriminant = v * v - 2.0 * g * h
    # Because h < peak, discriminant > 0.
    return 2.0 * math.sqrt(discriminant) / g


def timing_window(
    h: float, w: float, speed: float, config: Optional[Config] = None
) -> float:
    """Theoretical timing margin available for a jump over an obstacle.

    This is ``time_above(h)`` minus the horizontal overlap duration
    ``(w + PLAYER_WIDTH) / speed``.  A positive value means a perfect-timed jump
    can clear the obstacle; the usable offset budget is half of this margin.
    """
    cfg = config or Config()
    overlap = (w + cfg.PLAYER_WIDTH) / speed
    return time_above(h, cfg) - overlap


def clears_obstacle(
    h: float,
    w: float,
    speed: float,
    jump_offset_s: float,
    config: Optional[Config] = None,
) -> bool:
    """Theoretical clearance test for a single jump timed ``jump_offset_s`` from ideal.

    Ideal timing means the centre of the horizontal overlap interval is aligned
    with the jump apex.  A positive offset means the player jumped late.
    """
    cfg = config or Config()
    margin = timing_window(h, w, speed, cfg)
    if margin <= 0.0:
        return False
    allowed = 0.5 * margin
    return abs(jump_offset_s) <= allowed + 1e-12


def attempt_jump(
    player_y: float,
    velocity_y: float,
    ground_y: float,
    player_h: float,
    jump_velocity: float,
) -> tuple[float, bool]:
    """Pure-logic jump attempt.  Returns (new_velocity_y, accepted).

    The jump is accepted only when the player is on the ground.  This helper is
    used by the game and tested independently so the airborne-rejection rule is
    verified without a camera or display.
    """
    on_ground = player_y >= ground_y - player_h - 1
    if on_ground:
        return jump_velocity, True
    return velocity_y, False


# ---------------------------------------------------------------------------
# Spawn control
# ---------------------------------------------------------------------------

def spawn_allowed(
    prev_obstacle: Optional[Obstacle],
    new_left_x: float,
    speed: float,
    config: Optional[Config] = None,
) -> bool:
    """Check that a new obstacle respects the minimum physical gap.

    The gap is measured from the previous obstacle's right edge to the new
    obstacle's left edge at the moment of spawn.

    Derivation:
      * The player needs the full jump air time (about 0.767 s) to be safely
        airborne, plus an extra landing margin (0.35 s) before the next obstacle
        must be avoided.
      * While the spawn timer counts down, the previous obstacle moves left by
        ``speed * dt`` and the new obstacle spawns at ``SPAWN_X``.
      * Therefore the spawned gap is ``speed * dt - previous_width``.
      * We enforce ``gap >= speed * (AIR_TIME + LANDING_MARGIN)`` and also keep
        a floor of 350 px so the rule does not collapse at low difficulty.

    Obstacles that have already left the screen are ignored, because the player
    no longer cares about them for spacing.

    This does *not* guarantee success for all human latencies, camera buffers,
    or display latency; it is a theoretical lower bound.
    """
    cfg = config or Config()
    if prev_obstacle is None:
        return True
    # Ignore obstacles that have fully left the screen.
    if prev_obstacle.x + prev_obstacle.w < 0.0:
        return True
    prev_right = prev_obstacle.x + prev_obstacle.w
    gap_px = new_left_x - prev_right
    required = max(
        cfg.MIN_GAP_FLOOR_PX,
        speed * (air_time(cfg) + cfg.LANDING_MARGIN_S),
    )
    return gap_px >= required - 1e-9


def randomize_height(
    profile: DifficultyProfile, rng: random.Random, config: Optional[Config] = None
) -> float:
    cfg = config or Config()
    j = cfg.HEIGHT_JITTER
    return profile.height * rng.uniform(1.0 - j, 1.0 + j)




def try_spawn(
    prev_obstacle: Optional[Obstacle],
    profile: DifficultyProfile,
    speed: float,
    rng: random.Random,
    config: Optional[Config] = None,
    d: float = 0.0,
) -> Optional[Obstacle]:
    """Attempt to spawn a new obstacle respecting the min-gap rule.

    Returns the new ``Obstacle`` or ``None`` if the gap is too small (caller
    should retry on the next frame).
    """
    cfg = config or Config()
    left = cfg.SPAWN_X
    if not spawn_allowed(prev_obstacle, left, speed, cfg):
        return None
    h = randomize_height(profile, rng, cfg)
    w = profile.width  # no width jitter
    return Obstacle(
        x=float(left), w=w, h=h, done=False, failed=False,
        d_spawn=d, speed_spawn=profile.speed,
    )


def next_spawn_timer(
    profile: DifficultyProfile,
    rng: random.Random,
    config: Optional[Config] = None,
) -> float:
    """Return a jittered spawn timer in seconds.

    The interval is multiplied by a uniform factor in
    ``Config.SPAWN_INTERVAL_JITTER``.  The default range (1.0, 1.3) only ever
    lengthens the interval, so the shortest possible interval is the base
    ``profile.spawn_interval``.
    """
    cfg = config or Config()
    jitter_min, jitter_max = cfg.SPAWN_INTERVAL_JITTER
    return profile.spawn_interval * rng.uniform(jitter_min, jitter_max)


# ---------------------------------------------------------------------------
# Obstacle / outcome resolution
# ---------------------------------------------------------------------------

def _rects_overlap(
    ob: Obstacle, player_x: float, player_y: float, player_w: float, player_h: float, ground_y: float
) -> bool:
    """Pure rectangle overlap test between an obstacle and the player."""
    ob_left = ob.x
    ob_right = ob.x + ob.w
    ob_top = ground_y - ob.h
    ob_bottom = ground_y

    pl_left = player_x
    pl_right = player_x + player_w
    pl_top = player_y
    pl_bottom = player_y + player_h

    horizontal = ob_left < pl_right and ob_right > pl_left
    vertical = ob_top < pl_bottom and ob_bottom > pl_top
    return horizontal and vertical


def resolve_obstacles(
    obstacles: list[Obstacle],
    player_x: float,
    player_y: float,
    player_w: float,
    player_h: float,
    ground_y: float,
    speed: float,
    dt: float,
    controller: AdaptiveController,  # forward declaration handled by annotations
    config: Optional[Config] = None,
    on_outcome: Optional[Callable[[Obstacle, int], None]] = None,
) -> list[Obstacle]:
    """Move obstacles, record exactly one outcome per obstacle, and update ``d``.

    Success: obstacle's right edge passes the player's left edge with no
    collision ever.  Failure: first frame of overlap.  ``done`` prevents double
    counting.  Every obstacle (including ``done`` ones) keeps scrolling left.
    Off-screen obstacles are removed.
    """
    cfg = config or Config()

    def _record(ob: Obstacle, outcome: int) -> None:
        controller.update(outcome)
        if on_outcome is not None:
            on_outcome(ob, outcome)

    for ob in obstacles:
        # Failure on the first frame of overlap; only check live obstacles.
        if not ob.done and _rects_overlap(ob, player_x, player_y, player_w, player_h, ground_y):
            ob.failed = True
            ob.done = True
            _record(ob, 0)

        # Every obstacle keeps scrolling, even after it is done.
        ob.x -= speed * dt

        # Success once the obstacle has fully passed the player.
        if not ob.done and ob.x + ob.w < player_x:
            ob.done = True
            _record(ob, 1)

    # Remove obstacles that have fully left the screen.
    keep = [ob for ob in obstacles if ob.x + ob.w > -10.0]
    return keep


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class AdaptiveController:
    """Target-success-rate controller.

    The update rule is
        d <- clip(d + ETA * (outcome - P_TARGET))
    where outcome is 1 for success and 0 for failure.

    With the default ETA = 0.08 and P_TARGET = 0.75:
        success: d += +0.02
        crash:   d += -0.06

    The recent-success window is diagnostic only and never drives the update.
    """

    def __init__(
        self, config: Optional[Config] = None, rng: Optional[random.Random] = None
    ):
        self.config = config or Config()
        self.rng = rng or random.Random()
        self.d = self.config.D_START
        self.successes = 0
        self.failures = 0
        self.outcomes = 0
        self.recent = deque(maxlen=self.config.RECENT_WINDOW)

    def update(self, outcome: int) -> float:
        if outcome not in (0, 1):
            raise ValueError("outcome must be 0 (failure) or 1 (success)")
        cfg = self.config
        self.d = _clamp(
            self.d + cfg.ADAPTATION_RATE * (outcome - cfg.TARGET_SUCCESS_RATE),
            cfg.D_MIN,
            cfg.D_MAX,
        )
        if outcome:
            self.successes += 1
        else:
            self.failures += 1
        self.outcomes += 1
        self.recent.append(outcome)
        return self.d

    def reset(self) -> None:
        self.d = self.config.D_START
        self.successes = 0
        self.failures = 0
        self.outcomes = 0
        self.recent.clear()

    def recent_success_rate(self) -> float:
        if not self.recent:
            return 0.0
        return sum(self.recent) / len(self.recent)


# ---------------------------------------------------------------------------
# Latency diagnostics
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Rolling-window latency statistics.  Safe with empty or tiny samples."""

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.values: deque[float] = deque(maxlen=maxlen)

    def add(self, value: float) -> None:
        self.values.append(float(value))

    def __len__(self) -> int:
        return len(self.values)

    @property
    def count(self) -> int:
        return len(self.values)

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return statistics.mean(self.values)

    def median(self) -> float:
        if not self.values:
            return 0.0
        return statistics.median(self.values)

    def p90(self) -> float:
        if not self.values:
            return 0.0
        n = len(self.values)
        if n == 1:
            return self.values[0]
        sorted_vals = sorted(self.values)
        k = 0.9 * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return statistics.stdev(self.values)
