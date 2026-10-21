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

    # Which knob d turns.
    # "window" (default): d sets the width W of the safe jump window directly,
    #   on a log scale from WINDOW_EASY_S (d = 0) to WINDOW_HARD_S (d = 1):
    #       W(d) = WINDOW_EASY_S * (WINDOW_HARD_S / WINDOW_EASY_S) ** d
    #   The obstacle height is solved from W (jump_service.height_for_window),
    #   so difficulty is the timing precision the player needs, and every
    #   obstacle is feasible by construction.  Speed and width change only a
    #   little, because faster speed *widens* the clearance window.
    # "legacy": d scales speed, height, width and spawn interval linearly
    #   (SPEED_*, HEIGHT_*, WIDTH_*).  Kept for old logs and comparisons.  Its
    #   hardest setting is a ~240 ms window, which most players clear >90 % of
    #   the time, so a 75 % target pins d at 1.0 (see dev_bench "knob").
    DIFFICULTY_MODE: str = "window"     # "window" | "legacy"
    WINDOW_EASY_S: float = 0.45         # s, window at d = 0
    WINDOW_HARD_S: float = 0.05         # s, window at d = 1
    WINDOW_SPEED_MIN: float = 380.0     # px/s at d = 0
    WINDOW_SPEED_MAX: float = 460.0     # px/s at d = 1
    WINDOW_OBSTACLE_WIDTH: float = 30.0  # px, fixed
    # Per-obstacle variety, applied to the window (not the height):
    # W_obstacle = W(d) * exp(N(0, WINDOW_JITTER_STD)), kept feasible.
    WINDOW_JITTER_STD: float = 0.10

    # Legacy difficulty profile -- linear mapping from d in [0, 1] to game
    # parameters, used when DIFFICULTY_MODE == "legacy".
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

    # Spawn rhythm.  "jitter" = base interval x small normal jitter (nearly a
    # metronome).  "wild" = gap drawn uniformly between the shortest feasible
    # gap and a symmetric upper bound, so the rhythm is unpredictable.  In
    # "wild" mode the average interval is the profile interval, but never less
    # than (shortest feasible interval + SPAWN_WILD_ROOM_S): this keeps real
    # room for variation at high d, where the profile interval is already
    # close to the physical floor.
    SPAWN_GAP_MODE: str = "wild"        # "wild" | "jitter" | "probe"
    SPAWN_WILD_ROOM_S: float = 0.7      # s of average room above the floor
    # "probe" = measurement mode.  The rhythm is set directly by the REST time:
    # spawn-to-spawn gap = air time + U(REST_LO, REST_HI).  REST is the time
    # between landing and the ideal take-off for the next obstacle when the
    # first jump is perfectly timed, so REST < 0 is impossible without
    # early/late luck.  The profile interval and the min-gap floor are ignored
    # (only non-overlap is required), so gaps can be far tighter than normal.
    SPAWN_REST_LO_S: float = 0.0        # s
    SPAWN_REST_HI_S: float = 0.7        # s

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
    D_START: float = 0.3              # window mode: W ~ 235 ms, comfortable
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
# d values are for DIFFICULTY_MODE "window": starting windows of about
# 325 ms (Easy), 235 ms (Normal) and 170 ms (Hard).  The controller then
# moves d to where the player clears TARGET_SUCCESS_RATE.
# (The old legacy-mode presets were 0.80 / 0.94 / 1.00, at its ceiling.)
PRESETS = {
    1: {"d": 0.15, "threshold": 0.50, "stability": 1, "color": (120, 220, 120)},
    2: {"d": 0.30, "threshold": 0.60, "stability": 2, "color": (255, 200, 90)},
    3: {"d": 0.45, "threshold": 0.70, "stability": 3, "color": (255, 110, 110)},
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
    "speed", "height", "width", "outcome", "gap_s",
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
    window: float = 0.0   # target safe-window width (s); 0.0 in legacy mode


@dataclass
class Obstacle:
    x: float
    w: float
    h: float
    done: bool = False
    failed: bool = False
    d_spawn: float = 0.0
    speed_spawn: float = 0.0
    gap_s: float = 0.0   # time since the previous obstacle spawned (0 = none)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def difficulty_profile(d: float, config: Optional[Config] = None) -> DifficultyProfile:
    """Map normalized difficulty d in [0, 1] to game parameters.

    See ``Config.DIFFICULTY_MODE`` for what d means.
    """
    cfg = config or Config()
    d = _clamp(d, cfg.D_MIN, cfg.D_MAX)
    if cfg.DIFFICULTY_MODE == "window":
        return _window_profile(d, cfg)
    if cfg.DIFFICULTY_MODE != "legacy":
        raise ValueError(f"unknown DIFFICULTY_MODE {cfg.DIFFICULTY_MODE!r}")
    speed = cfg.SPEED_MIN + (cfg.SPEED_MAX - cfg.SPEED_MIN) * d
    height = cfg.HEIGHT_MIN + (cfg.HEIGHT_MAX - cfg.HEIGHT_MIN) * d
    width = cfg.WIDTH_MIN + (cfg.WIDTH_MAX - cfg.WIDTH_MIN) * d
    spawn_interval = cfg.SPAWN_INTERVAL_BASE_AT_D0 - (
        cfg.SPAWN_INTERVAL_BASE_AT_D0 - cfg.SPAWN_INTERVAL_BASE_AT_D1
    ) * d
    return DifficultyProfile(speed, height, width, spawn_interval)


def window_for_d(d: float, config: Optional[Config] = None) -> float:
    """Target safe-window width (s) for d in window mode (log scale)."""
    cfg = config or Config()
    d = _clamp(d, cfg.D_MIN, cfg.D_MAX)
    return cfg.WINDOW_EASY_S * (cfg.WINDOW_HARD_S / cfg.WINDOW_EASY_S) ** d


def d_for_window(window_s: float, config: Optional[Config] = None) -> float:
    """Inverse of ``window_for_d`` (clamped to [D_MIN, D_MAX])."""
    cfg = config or Config()
    d = math.log(window_s / cfg.WINDOW_EASY_S) / math.log(
        cfg.WINDOW_HARD_S / cfg.WINDOW_EASY_S
    )
    return _clamp(d, cfg.D_MIN, cfg.D_MAX)


def height_for_target_window(
    window_s: float, width: float, speed: float, config: Optional[Config] = None
) -> float:
    """Obstacle height (px) whose game-scored window is ``window_s`` seconds.

    Uses the frame-corrected jump from ``experiment.game_physics``, the same
    physics experiment mode uses, so the window is what the game really scores.
    """
    # Imported here: both modules import this one.
    from gesture_lab import jump_service as js
    from gesture_lab.experiment import game_physics

    cfg = config or Config()
    return js.height_for_window(window_s, width, speed, game_physics(cfg))


def _window_profile(d: float, cfg: Config) -> DifficultyProfile:
    window = window_for_d(d, cfg)
    speed = cfg.WINDOW_SPEED_MIN + (cfg.WINDOW_SPEED_MAX - cfg.WINDOW_SPEED_MIN) * d
    width = cfg.WINDOW_OBSTACLE_WIDTH
    height = height_for_target_window(window, width, speed, cfg)
    spawn_interval = cfg.SPAWN_INTERVAL_BASE_AT_D0 - (
        cfg.SPAWN_INTERVAL_BASE_AT_D0 - cfg.SPAWN_INTERVAL_BASE_AT_D1
    ) * d
    return DifficultyProfile(speed, height, width, spawn_interval, window)


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


def fixed_steps(accum: float, frame_s: float, step_s: float) -> tuple[int, float]:
    """TIMING: split elapsed real time into whole fixed physics steps.

    The camera sets how often the game loop runs (often about 30 frames per
    second), but the jump and obstacle windows are only valid for a fixed
    physics step.  Each loop pass adds its real duration ``frame_s`` to the
    carried-over ``accum`` and runs as many steps of ``step_s`` as fit.
    Returns ``(steps_to_run, new_accum)``; the remainder is kept for the next
    pass, so game time never drifts from real time by more than one step.
    """
    total = accum + frame_s
    n = int(total / step_s + 1e-6)
    return n, max(0.0, total - n * step_s)


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
    if cfg.SPAWN_GAP_MODE == "probe":
        # Measurement mode: spacing is controlled by the rest-time sampler, so
        # only forbid overlapping obstacles.
        return gap_px >= 0.0
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
    if profile.window > 0.0:
        # Window mode: jitter the window, not the height.  Near the jump apex a
        # few pixels of height change the window a lot, so height jitter would
        # make the hardest obstacles random (or impossible).
        lo = 0.8 * cfg.WINDOW_HARD_S
        hi = window_for_d(cfg.D_MIN, cfg)
        w = profile.window * math.exp(rng.gauss(0.0, cfg.WINDOW_JITTER_STD))
        w = _clamp(w, lo, hi)
        return height_for_target_window(w, profile.width, profile.speed, cfg)
    multiplier = _sample_jitter_multiplier(
        rng, cfg.HEIGHT_JITTER_MEAN, cfg.HEIGHT_JITTER_STD
    )
    return profile.height * multiplier


def design_margin_height(
    profile: DifficultyProfile, config: Optional[Config] = None
) -> float:
    """Height (px) of a +2 sigma 'hard draw' obstacle, for design tables.

    Legacy mode jitters height, so the hard draw is ``height x (mean + 2 std)``.
    Window mode jitters the window, so the hard draw is the window
    ``W x exp(-2 std)`` (kept at or above the same floor as ``randomize_height``).
    """
    cfg = config or Config()
    if profile.window > 0.0:
        w = max(0.8 * cfg.WINDOW_HARD_S,
                profile.window * math.exp(-2.0 * cfg.WINDOW_JITTER_STD))
        return height_for_target_window(w, profile.width, profile.speed, cfg)
    return profile.height * (cfg.HEIGHT_JITTER_MEAN + 2.0 * cfg.HEIGHT_JITTER_STD)


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


def min_spawn_interval(
    profile: DifficultyProfile, config: Optional[Config] = None
) -> float:
    """Shortest spawn-to-spawn time (s) that satisfies the min-gap rule.

    This is ``spawn_allowed`` expressed in seconds: the obstacle must travel
    the required gap plus the previous obstacle's width.  The previous width is
    approximated by the current profile width.  The game's own spawn check
    still has the last word, so an approximation here can never produce an
    infeasible pair.
    """
    cfg = config or Config()
    required_px = max(
        cfg.MIN_GAP_FLOOR_PX,
        profile.speed * (air_time(cfg) + cfg.LANDING_MARGIN_S),
    )
    return (required_px + profile.width) / profile.speed


def next_spawn_timer(
    profile: DifficultyProfile,
    rng: random.Random,
    config: Optional[Config] = None,
) -> float:
    """Return the time (s) until the next obstacle should spawn.

    ``SPAWN_GAP_MODE == "wild"``: uniform on ``[lo, 2*mean - lo]`` where ``lo``
    is the shortest feasible interval and ``mean`` is the profile interval
    (raised to ``lo + SPAWN_WILD_ROOM_S`` if it is closer to the floor than
    that).  The mean is preserved, no gap is below the feasible minimum, and no
    gap is more than about twice the mean.

    ``SPAWN_GAP_MODE == "probe"``: ``air_time + U(SPAWN_REST_LO_S,
    SPAWN_REST_HI_S)``, i.e. the rest time is sampled directly and may be far
    below the normal feasibility floor (measurement only).

    ``SPAWN_GAP_MODE == "jitter"``: the original behaviour, described below.

    Original jitter behaviour:

    The interval is multiplied by a sample from
    ``N(Config.SPAWN_INTERVAL_JITTER_MEAN, Config.SPAWN_INTERVAL_JITTER_STD)``.
    The positive floor in sampling keeps the timer strictly positive; the
    caller's min-gap check prevents impossibly short spawns.
    """
    cfg = config or Config()
    if cfg.SPAWN_GAP_MODE == "wild":
        lo = min_spawn_interval(profile, cfg)
        mean = max(profile.spawn_interval, lo + cfg.SPAWN_WILD_ROOM_S)
        return rng.uniform(lo, 2.0 * mean - lo)
    if cfg.SPAWN_GAP_MODE == "probe":
        return air_time(cfg) + rng.uniform(cfg.SPAWN_REST_LO_S, cfg.SPAWN_REST_HI_S)
    if cfg.SPAWN_GAP_MODE != "jitter":
        raise ValueError(
            "SPAWN_GAP_MODE must be 'wild', 'jitter' or 'probe', "
            f"got {cfg.SPAWN_GAP_MODE!r}"
        )
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
    per_obstacle_speed: bool = False,
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
        # per_obstacle_speed: each obstacle keeps the speed it spawned with (speed ladder).
        ob.x -= (ob.speed_spawn if per_obstacle_speed and ob.speed_spawn > 0 else speed) * dt

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
