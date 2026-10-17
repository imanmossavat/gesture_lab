"""Pure-Python adaptive difficulty logic for the gesture runner.

No pygame, cv2, or mediapipe imports.  This module is shared by the game,
the dev bench, and the tests.
"""
from __future__ import annotations

import copy
import math
import random
import statistics
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, NamedTuple, Optional


@dataclass(frozen=True)
class Config:
    """Every tunable number for the adaptive difficulty system.

    The physics constants (GRAVITY, JUMP_VELOCITY) are *measurements* taken from
    the existing game.py.  All other values are *theoretical design choices* and
    must be treated as guesses until they are validated with real players.
    """

    # World / player geometry (px)
    PLAYER_X: float = 620.0
    PLAYER_WIDTH: float = 42.0
    PLAYER_HEIGHT: float = 56.0
    GROUND_Y: float = 590.0
    SPAWN_X: float = 1190.0  # WIDTH + 10 from game.py (px)

    # Physics constants -- measured from the existing game file.
    GRAVITY: float = 1800.0          # px/s^2 (downward positive)
    JUMP_VELOCITY: float = -690.0     # px/s (upward negative)

    # Difficulty profile -- theoretical mapping from d in [0, 1] to game parameters.
    # These six fields are the knobs that control how hard the game gets.
    # If players keep pinning d at 1.0, raise the *_MAX values (and keep
    # feasibility by checking timing_window() / spawn_allowed()).
    SPEED_MIN: float = 260.0          # px/s
    SPEED_MAX: float = 550.0          # px/s - hardest obstacle speed at d=1
    HEIGHT_MIN: float = 30.0          # px
    HEIGHT_MAX: float = 90.0          # px - tallest obstacle at d=1
    WIDTH_MIN: float = 20.0           # px
    WIDTH_MAX: float = 50.0           # px - widest obstacle at d=1
    SPAWN_INTERVAL_BASE_AT_D0: float = 2.5   # s
    SPAWN_INTERVAL_BASE_AT_D1: float = 1.3   # s (2.5 - 1.2)

    # Spawn interval jitter -- multiplicative, sampled from N(mean, std).
    # The mean of 1.0 keeps the average interval at the base value; a small
    # std adds normal variation while the positive floor in sampling prevents
    # non-positive intervals.
    SPAWN_INTERVAL_JITTER_MEAN: float = 1.0
    SPAWN_INTERVAL_JITTER_STD: float = 0.07

    # Height jitter -- multiplicative, sampled from N(mean, std).
    # Only height is jittered; width is fixed so the min-gap guarantee holds.
    HEIGHT_JITTER_MEAN: float = 1.0
    HEIGHT_JITTER_STD: float = 0.07

    # Controller -- theoretical choices, not calibrated.
    TARGET_SUCCESS_RATE: float = 0.75
    ADAPTATION_RATE: float = 0.08
    # Momentum (beta) in [0, 1). 0.0 = original memoryless update. Independent of
    # ADAPTATION_RATE (the learning rate): each step is added to a running
    # velocity that decays by beta, so streaks accelerate d and single outcomes
    # are smoothed. Long-run gain is ADAPTATION_RATE / (1 - MOMENTUM).
    MOMENTUM: float = 0.0
    D_MIN: float = 0.0
    D_MAX: float = 1.0
    D_START: float = 0.5
    RECENT_WINDOW: int = 200

    # Minimum gap -- theoretical safety margin.
    LANDING_MARGIN_S: float = 0.35    # s
    MIN_GAP_FLOOR_PX: float = 350.0   # px

    # Latency diagnostics
    LATENCY_WINDOW: int = 200


class Phase(Enum):
    READY = "ready"
    PLAYING = "playing"
    PAUSED = "paused"
    OVER = "over"


def next_phase(phase: Phase, event: str) -> Phase:
    """Pure state transition for the game loop."""
    if event == "space":
        if phase == Phase.READY:
            return Phase.PLAYING
        if phase == Phase.PLAYING:
            return Phase.PAUSED
        if phase == Phase.PAUSED:
            return Phase.PLAYING
        if phase == Phase.OVER:
            return Phase.PLAYING
    if event == "game_over" and phase == Phase.PLAYING:
        return Phase.OVER
    if event == "preset":
        return Phase.READY
    return phase


# Presets only affect the starting d, classifier threshold, stability
# requirement, and UI colour.  All runtime parameters come from
# difficulty_profile(d).
# d values are centred on the observed ceiling (0.94) so the default
# Normal preset starts near the top of the adaptive range, with Easy
# slightly below and Hard clamped at the absolute ceiling.
PRESETS = {
    1: {"d": 0.80, "threshold": 0.50, "stability": 1, "color": (120, 220, 120)},
    2: {"d": 0.94, "threshold": 0.60, "stability": 2, "color": (255, 200, 90)},
    3: {"d": 1.00, "threshold": 0.70, "stability": 3, "color": (255, 110, 110)},
}


def build_presets(start_d=None, preset_id=2, base=PRESETS):
    """Return a deep copy of PRESETS with an optional initial-d override."""
    presets = copy.deepcopy(base)
    if start_d is not None:
        presets[preset_id]["d"] = start_d
    return presets


# CSV column order used by game.py --log-outcomes and dev_bench.py.
OUTCOME_HEADER = [
    "t_s", "player", "session", "mode", "d_spawn", "d_after",
    "speed", "height", "width", "outcome",
]

# CSV column order used by game.py --log-latency.
LATENCY_HEADER = [
    "frame_read_to_jump_ms", "first_open_to_jump_ms",
    "inference_ms", "loop_ms", "accepted",
]


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


def _sample_jitter_multiplier(
    rng: random.Random, mean: float, std: float, floor: float = 0.1
) -> float:
    """Sample a positive multiplier from a normal distribution.

    Values below ``floor`` are rejected and resampled, keeping the multiplier
    strictly positive.  With the default small std this rejection is extremely
    rare.
    """
    while True:
        v = rng.gauss(mean, std)
        if v >= floor:
            return v


def randomize_height(
    profile: DifficultyProfile, rng: random.Random, config: Optional[Config] = None
) -> float:
    """Return a jittered obstacle height in pixels.

    The base ``profile.height`` is multiplied by a sample from
    ``N(Config.HEIGHT_JITTER_MEAN, Config.HEIGHT_JITTER_STD)``.  The sample is
    rejected if it would be non-positive, so the returned height is always
    strictly positive.
    """
    cfg = config or Config()
    multiplier = _sample_jitter_multiplier(
        rng, cfg.HEIGHT_JITTER_MEAN, cfg.HEIGHT_JITTER_STD
    )
    return profile.height * multiplier


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

    The interval is multiplied by a sample from
    ``N(Config.SPAWN_INTERVAL_JITTER_MEAN, Config.SPAWN_INTERVAL_JITTER_STD)``.
    The positive floor in sampling keeps the timer strictly positive; the
    caller's min-gap check prevents impossibly short spawns.
    """
    cfg = config or Config()
    multiplier = _sample_jitter_multiplier(
        rng, cfg.SPAWN_INTERVAL_JITTER_MEAN, cfg.SPAWN_INTERVAL_JITTER_STD
    )
    return profile.spawn_interval * multiplier


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
        v <- BETA * v + ETA * (outcome - P_TARGET)
        d <- clip(d + v)
    where outcome is 1 for success and 0 for failure, ETA is the learning rate
    (ADAPTATION_RATE) and BETA is MOMENTUM.  The two are independent
    hyperparameters; with momentum a steady streak moves d by about
    ETA / (1 - BETA) per step, so lower ETA when raising BETA.
    If d hits a bound, v is zeroed so the velocity cannot wind up against it.

    With the default BETA = 0 (no momentum), ETA = 0.08 and P_TARGET = 0.75:
        success: d += +0.02
        crash:   d += -0.06

    The recent-success window is diagnostic only and never drives the update.
    """

    def __init__(
        self, config: Optional[Config] = None, rng: Optional[random.Random] = None
    ):
        self.config = config or Config()
        self.rng = rng or random.Random()
        self.d = _clamp(
            self.config.D_START, self.config.D_MIN, self.config.D_MAX
        )
        self.velocity = 0.0
        self.successes = 0
        self.failures = 0
        self.outcomes = 0
        self.recent = deque(maxlen=self.config.RECENT_WINDOW)

    def update(self, outcome: int) -> float:
        if outcome not in (0, 1):
            raise ValueError("outcome must be 0 (failure) or 1 (success)")
        cfg = self.config
        step = cfg.ADAPTATION_RATE * (outcome - cfg.TARGET_SUCCESS_RATE)
        self.velocity = cfg.MOMENTUM * self.velocity + step
        proposed = self.d + self.velocity
        self.d = _clamp(proposed, cfg.D_MIN, cfg.D_MAX)
        if self.d != proposed:  # hit a bound: stop velocity winding up
            self.velocity = 0.0
        if outcome:
            self.successes += 1
        else:
            self.failures += 1
        self.outcomes += 1
        self.recent.append(outcome)
        return self.d

    def reset(self) -> None:
        self.d = _clamp(
            self.config.D_START, self.config.D_MIN, self.config.D_MAX
        )
        self.velocity = 0.0
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


class SessionTracker:
    """Monotonic session counter for logged gameplay segments.

    A new session is started every time the player changes the difficulty
    preset or restarts the game.  The counter is intentionally simple so it
    can be unit-tested independently of the event loop.
    """

    def __init__(self, start: int = 0):
        self._session = start

    def bump(self) -> int:
        """Advance to the next session and return its id."""
        self._session += 1
        return self._session

    @property
    def current(self) -> int:
        return self._session
