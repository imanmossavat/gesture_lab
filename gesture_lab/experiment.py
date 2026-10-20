"""Experiment mode: a designed list of obstacles, and a recorder for jump timing.

No pygame, cv2, or mediapipe imports, so everything here can be unit-tested and
read on its own.

The idea
--------
The adaptive game asks "how hard should the next obstacle be?" and answers with
one number, d.  Here we ask a different question: "how precisely does THIS
player time a jump?"  To answer it we

1. choose a list of obstacles whose *timing windows* range from wide (easy) to
   narrow (hard)           -> ``make_trials`` / ``ExperimentPlan``
2. record, for every obstacle, WHEN the player jumped and what the physics says
   the allowed jump times were                -> ``TrialRecorder``

Then ``analyze_trials.py`` turns the log into answers.

The clock
---------
Everything is measured on one clock per obstacle:

    t = 0   the moment the obstacle's front edge reaches the player's front edge
    t < 0   before that moment (the obstacle is still approaching)

A jump (takeoff) at time ``t_j`` clears the obstacle only if ``L <= t_j <= U``;
see ``jump_service.success_window`` for where L and U come from.
"""
from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Optional, Sequence

from gesture_lab.adaptive import Config, Obstacle
from gesture_lab import jump_service as js


# ---------------------------------------------------------------------------
# 0. Physics as the game actually plays it
# ---------------------------------------------------------------------------

FRAME_DT = 1.0 / 60.0   # the game targets 60 frames per second


def game_physics(cfg: Config, dt: float = FRAME_DT) -> Config:
    """Return a Config whose jump matches the game's frame-by-frame update.

    The textbook jump is  y(t) = v0 t - g t^2 / 2.  The game does not move
    continuously.  Every frame it first updates the speed, then the position:

        v <- v - g dt          y <- y + v dt

    Summing this over n frames (t = n dt) gives

        y = v0 t - g t^2 / 2 - (g dt / 2) t  =  (v0 - g dt / 2) t - g t^2 / 2

    That is the textbook jump with a slightly smaller launch speed,
    ``v0 - g dt / 2``.  At 60 fps with g = 1800 px/s^2 this costs 15 px/s and
    lowers the highest point from 132.3 px to 126.5 px.  For tall obstacles
    near the apex that difference decides whether a jump clears, so all window
    calculations in experiment mode use this corrected jump.

    (We found this because the simulated player kept crashing on tall obstacles
    that the continuous physics said it should clear.  That is what the
    ``unexplained`` failure mode is for.)
    """
    # JUMP_VELOCITY is negative (up is negative y), so adding moves it toward 0.
    return dataclasses.replace(
        cfg, JUMP_VELOCITY=cfg.JUMP_VELOCITY + 0.5 * cfg.GRAVITY * dt
    )


def game_window(
    h: float, w: float, speed: float, gap_px: Optional[float],
    cfg: Config, dt: float = FRAME_DT,
) -> js.JumpWindow:
    """The success window [L, U] as the GAME scores it.

    Two corrections to the textbook window, both caused by how the game loop
    works frame by frame:

    1. The jump follows the frame-by-frame update (see ``game_physics``).
    2. ``resolve_obstacles`` checks for a collision BEFORE it moves the
       obstacles.  So in every frame the player has already been updated while
       the obstacle is still where it was one frame ago: the obstacle is
       effectively one frame late.  Seen from the obstacle's clock, the player's
       jump happened ``dt`` earlier, so every jump time that works is ``dt``
       later than the textbook says:

           L_game = L + dt      U_game = U + dt

    The shift is one frame (about 17 ms), which is small, but our narrowest
    windows are only 60 ms wide.
    """
    base = js.success_window(h, w, speed, gap_px, game_physics(cfg, dt))
    if not base.feasible:
        return base
    return dataclasses.replace(
        base, L=base.L + dt, U_clear=base.U_clear + dt,
        U_gap=base.U_gap + dt, U=base.U + dt,
    )


# ---------------------------------------------------------------------------
# 1. The designed list of obstacles
# ---------------------------------------------------------------------------

# Target window widths in seconds, widest (easy) to narrowest (hard).
# A window of 0.45 s means the player may jump anywhere inside a 450 ms span;
# 0.06 s means the jump must land inside a 60 ms span.
DEFAULT_WINDOWS = (0.45, 0.32, 0.24, 0.18, 0.13, 0.09, 0.06)

# Windows below this width count as "hard" when we check the order is safe.
HARD_BELOW_S = 0.13
MAX_HARD_IN_A_ROW = 2


@dataclass(frozen=True)
class Trial:
    """One planned obstacle."""
    index: int                       # position in the session, starting at 0
    target_w: float                  # the window width we aimed for (s)
    height: float                    # obstacle height (px)
    width: float                     # obstacle width (px)
    speed: float                     # game speed (px/s)
    interval_after: Optional[float]  # seconds until the next obstacle spawns
    gap_after_px: Optional[float]    # gap between this obstacle and the next (px)


def _is_safe_order(windows: Sequence[float]) -> bool:
    """True when no more than MAX_HARD_IN_A_ROW hard windows follow each other."""
    streak = 0
    for w in windows:
        streak = streak + 1 if w < HARD_BELOW_S else 0
        if streak > MAX_HARD_IN_A_ROW:
            return False
    return True


def _shuffled_blocks(
    windows: Sequence[float], blocks: int, rng: random.Random,
) -> list[float]:
    """Shuffle each block of windows, so every block uses every window once.

    Why blocks?  Pure random order could give the hardest window five times and
    the easiest never.  Blocks guarantee a balanced design while the order stays
    unpredictable.  We retry a shuffle until the order is "safe" (no long streak
    of hard obstacles) and until the same window does not repeat back to back.
    """
    order: list[float] = []
    for _ in range(blocks):
        for _attempt in range(200):
            block = list(windows)
            rng.shuffle(block)
            candidate = order + block
            repeats = order and block[0] == order[-1]
            if _is_safe_order(candidate) and not repeats:
                order = candidate
                break
        else:  # extremely unlikely; keep the last shuffle rather than loop forever
            order = order + block
    return order


def make_trials(
    cfg: Config,
    rng: random.Random,
    speed: float = 400.0,
    width: float = 30.0,
    windows: Sequence[float] = DEFAULT_WINDOWS,
    blocks: int = 4,
    warmup: int = 3,
    window_jitter: float = 0.05,
    interval_range: tuple[float, float] = (1.8, 2.6),
) -> list[Trial]:
    """Build the full, fixed list of obstacles for one session.

    * The first ``warmup`` obstacles use the widest window (a safe start).
    * After that come ``blocks`` shuffled blocks, one obstacle per window each.
    * Each target window is nudged by a random +-``window_jitter`` (5 %) so the
      same height does not repeat exactly.
    * The time between obstacles is random in ``interval_range`` seconds.  It is
      long enough that the gap never limits the jump: only the obstacle's own
      window matters (see ``check_gap_is_not_binding`` in the tests).
    """
    physics = game_physics(cfg)
    widest = max(windows)
    targets = [widest] * warmup + _shuffled_blocks(windows, blocks, rng)

    # Choose spacing first, because the gap after an obstacle is part of its row.
    intervals = [rng.uniform(*interval_range) for _ in targets]

    trials = []
    last = len(targets) - 1
    for i, target in enumerate(targets):
        jittered = target * (1.0 + rng.uniform(-window_jitter, window_jitter))
        height = js.height_for_window(jittered, width, speed, physics)
        is_last = i == last
        interval = None if is_last else intervals[i]
        gap = None if is_last else speed * interval - width
        trials.append(Trial(
            index=i, target_w=target, height=height, width=width,
            speed=speed, interval_after=interval, gap_after_px=gap,
        ))
    return trials


class ExperimentPlan:
    """Hands out the planned trials one at a time."""

    def __init__(self, trials: Sequence[Trial]):
        self._trials = list(trials)
        self._next = 0

    @property
    def total(self) -> int:
        return len(self._trials)

    @property
    def spawned(self) -> int:
        return self._next

    @property
    def exhausted(self) -> bool:
        return self._next >= len(self._trials)

    def take_next(self) -> Optional[Trial]:
        """Return the next trial, or None when the list is used up."""
        if self.exhausted:
            return None
        trial = self._trials[self._next]
        self._next += 1
        return trial


def trial_to_obstacle(trial: Trial, cfg: Config) -> Obstacle:
    """Create the game obstacle for a planned trial."""
    return Obstacle(
        x=float(cfg.SPAWN_X), w=trial.width, h=trial.height,
        d_spawn=0.0, speed_spawn=trial.speed,
    )


# ---------------------------------------------------------------------------
# 2. Recording the jump time of every obstacle
# ---------------------------------------------------------------------------

# One column per fact we want to analyse later.  Order matters: it is the CSV
# column order.
TRIAL_HEADER = [
    "t_s",                # game time when the outcome was decided (s)
    "player", "session",
    "trial",              # obstacle number in this session (0, 1, 2, ...)
    "target_w",           # window width we aimed for (s); empty outside experiment mode
    "height", "width", "speed",   # obstacle geometry (px) and game speed (px/s)
    "next_gap_px",        # gap to the following obstacle (px); empty if unknown
    "L", "U", "W",        # the physics window (s): earliest, latest, width
    "binding",            # which limit sets U: "clearance" or "gap"
    "t_j",                # when the player jumped (s on the obstacle clock); empty if no jump
    "delta",              # t_j minus the window centre (s); 0 = perfectly centred
    "jumped",             # 1 if a jump started before the obstacle reached the player
    "predicted_success",  # 1 if L <= t_j <= U (what the physics predicts)
    "outcome",            # 1 = cleared, 0 = crashed (what the game decided)
    "failure_mode",       # early / late / no_jump / unexplained (empty on success)
    # TIMING: two clocks.  t_j is when the GAME started the jump (it decides clear or crash).
    # t_open is when the camera first SAW the gesture, minus the average wait for the
    # next camera frame: your timing with the software's own delay removed.  What is
    # left in it is the camera's internal delay and your movement time (not measurable here).
    "open_lag_ms",        # first frame that showed the gesture -> jump applied (ms); empty if unknown
    "cam_interval_ms",    # camera frame interval when you jumped (ms); empty if unknown
    "t_open",             # t_j - open_lag - cam_interval / 2 (s)
    "delta_open",         # t_open minus the window centre (s); the number to read for YOUR timing
]


@dataclass
class _LiveTrial:
    """An obstacle that is still on its way, plus what we know about it."""
    obstacle: Obstacle
    index: int
    target_w: Optional[float]
    next_gap_px: Optional[float]
    t_j: Optional[float] = None
    lag_s: Optional[float] = None       # first gesture frame -> jump applied (s)
    cam_s: Optional[float] = None       # camera frame interval (s)


class TrialRecorder:
    """Watches obstacles and jumps, and produces one log row per obstacle.

    Usage inside the game loop (three calls):

        recorder.register(obstacle, ...)             when an obstacle spawns
        recorder.note_jump(speed)                    when a jump is ACCEPTED
        row = recorder.finish(obstacle, outcome, ...) when the outcome is decided
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._live: list[_LiveTrial] = []
        self._count = 0

    # -- step 1: an obstacle appears -------------------------------------
    def register(
        self, obstacle: Obstacle,
        target_w: Optional[float] = None,
        next_gap_px: Optional[float] = None,
        index: Optional[int] = None,
    ) -> None:
        if index is None:
            index = self._count
        self._count += 1
        # The gap to the NEXT obstacle is only known once that obstacle spawns.
        # If the previous obstacle is still on its way and had no planned gap
        # (adaptive modes), fill it in now: spawn position minus the previous
        # obstacle's right edge.  Matters when gaps are tight (probe mode).
        if self._live and self._live[-1].next_gap_px is None:
            prev = self._live[-1].obstacle
            self._live[-1].next_gap_px = max(0.0, obstacle.x - (prev.x + prev.w))
        self._live.append(_LiveTrial(obstacle, index, target_w, next_gap_px))

    # -- step 2: the player jumps ----------------------------------------
    def note_jump(
        self, speed: float, per_obstacle: bool = False,
        lag_s: Optional[float] = None, cam_interval_s: Optional[float] = None,
    ) -> Optional[float]:
        """Remember when this jump happened, for the obstacle it belongs to.

        A jump belongs to the nearest obstacle that has NOT yet reached the
        player's front edge.  If the player jumps twice before an obstacle
        arrives, the later jump replaces the earlier one (only the last takeoff
        before the obstacle reaches the player matters).

        The time is measured from the obstacle's current position:

            t_j = -(distance from obstacle front to player front) / speed

        which is negative while the obstacle is still approaching.
        Returns t_j, or None if no obstacle is ahead.
        """
        player_front = self.cfg.PLAYER_X + self.cfg.PLAYER_WIDTH
        ahead = [lt for lt in self._live if lt.obstacle.x > player_front]
        if not ahead or speed <= 0.0:
            return None
        nearest = min(ahead, key=lambda lt: lt.obstacle.x)
        if per_obstacle and nearest.obstacle.speed_spawn > 0:
            speed = nearest.obstacle.speed_spawn     # speed ladder: use the obstacle's own speed
        t_j = -(nearest.obstacle.x - player_front) / speed
        nearest.t_j = t_j
        nearest.lag_s, nearest.cam_s = lag_s, cam_interval_s   # for t_open (see TRIAL_HEADER)
        return t_j

    # -- step 3: the game decides the outcome ----------------------------
    def finish(
        self, obstacle: Obstacle, outcome: int, play_time: float,
        player: str = "anon", session: int = 0,
    ) -> Optional[dict]:
        """Turn one finished obstacle into a log row (a dict keyed by TRIAL_HEADER)."""
        live = next((lt for lt in self._live if lt.obstacle is obstacle), None)
        if live is None:
            return None
        self._live.remove(live)

        speed = obstacle.speed_spawn
        window = game_window(
            obstacle.h, obstacle.w, speed, live.next_gap_px, self.cfg
        )
        t_j = live.t_j
        jumped = t_j is not None
        delta = (t_j - window.centre) if jumped and window.feasible else None
        predicted = int(jumped and window.L <= t_j <= window.U)
        corrected = jumped and live.lag_s is not None and live.cam_s is not None
        t_open = (t_j - live.lag_s - live.cam_s / 2.0) if corrected else None
        delta_open = (t_open - window.centre) if corrected and window.feasible else None

        return {
            "t_s": f"{play_time:.3f}",
            "player": player,
            "session": session,
            "trial": live.index,
            "target_w": _fmt(live.target_w, 3),
            "height": f"{obstacle.h:.1f}",
            "width": f"{obstacle.w:.1f}",
            "speed": f"{speed:.1f}",
            "next_gap_px": _fmt(live.next_gap_px, 1),
            "L": _fmt(window.L if window.feasible else None, 4),
            "U": _fmt(window.U if window.feasible else None, 4),
            "W": _fmt(window.width if window.feasible else None, 4),
            "binding": ("gap" if window.U_gap < window.U_clear else "clearance")
                       if window.feasible else "",
            "t_j": _fmt(t_j, 4),
            "delta": _fmt(delta, 4),
            "jumped": int(jumped),
            "predicted_success": predicted,
            "outcome": int(outcome),
            "failure_mode": classify_failure(outcome, t_j, window),
            "open_lag_ms": _fmt(live.lag_s * 1000 if corrected else None, 1),
            "cam_interval_ms": _fmt(live.cam_s * 1000 if corrected else None, 1),
            "t_open": _fmt(t_open, 4),
            "delta_open": _fmt(delta_open, 4),
        }


def classify_failure(outcome: int, t_j: Optional[float], window: js.JumpWindow) -> str:
    """Say WHY an obstacle was a failure, using the physics window.

    ``early``       jumped before L: came down on the obstacle
    ``late``        jumped after U: hit the obstacle on the way up
    ``no_jump``     no jump started before the obstacle reached the player
    ``unexplained`` the physics says this jump should have worked, but the game
                    crashed anyway.  These rows are the check on the model: if
                    there are many, the window formula or its inputs are wrong.
    """
    if outcome:
        return ""
    if t_j is None:
        return "no_jump"
    if not window.feasible:
        return "unexplained"
    if t_j < window.L:
        return "early"
    if t_j > window.U:
        return "late"
    return "unexplained"


def _fmt(value: Optional[float], digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def row_values(row: dict) -> list:
    """Row dict -> list in TRIAL_HEADER order, ready for ``csv.writer``."""
    return [row[name] for name in TRIAL_HEADER]
