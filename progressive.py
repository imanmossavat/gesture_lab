"""A progressive session: a ladder of speeds, and on each rung a staircase that
homes in on the narrowest safe window the player can handle.

Why: the fixed experiment list measures the player; this plan *plays* the player.
It keeps pushing toward the edge of what is feasible for them, and then raises the
speed.  It is deliberately not a rigid pattern:

* ``track`` obstacles follow the staircase (window shrinks after a clear, grows
  faster after a crash, so it settles where the player clears ``target_rate``).
* ``relief`` obstacles are easy (twice the current window) and come at random, and
  always after two crashes in a row, so a bad patch never snowballs.
* ``stretch`` obstacles are harder (0.7 x the current window) at random.
Only ``track`` obstacles move the staircase.  All windows are feasible by construction
(the height comes from the physics, see ``jump_service.height_for_window``).

Same interface as ``experiment.ExperimentPlan`` (``total``, ``spawned``, ``exhausted``,
``take_next``) plus ``report(index, cleared)``, which the game calls for every outcome.
Pure logic: no pygame, no camera.
"""
from __future__ import annotations

import math
import random
from typing import Optional

import jump_service as js
from adaptive import Config
from experiment import Trial, game_physics

W_MIN, W_MAX = 0.05, 0.45     # seconds: 50 ms is hard but feasible, 450 ms is easy
STEP_DOWN = 0.07              # log-step after a clear (window shrinks about 7 %)
P_RELIEF, P_STRETCH = 0.20, 0.10


class ProgressivePlan:
    """Hands out obstacles one at a time, adapting to the player's outcomes."""

    def __init__(
        self, cfg: Config, rng: random.Random, speed0: float = 400.0,
        speed_step: float = 60.0, levels: int = 3, per_level: int = 10,
        warmup: int = 3, target_rate: float = 0.75, width: float = 30.0,
        start_w: float = 0.32, interval_range: tuple[float, float] = (1.8, 2.6),
    ):
        self.cfg, self.rng = cfg, rng
        self.speed0, self.speed_step = speed0, speed_step
        self.levels, self.per_level, self.warmup = levels, per_level, warmup
        self.width, self.interval_range = width, interval_range
        # Weighted up/down staircase: it settles where  p * down = (1 - p) * up.
        self.step_up = STEP_DOWN * target_rate / (1.0 - target_rate)
        self.w = start_w                 # current threshold estimate (s)
        self._next = 0
        self._kinds: dict[int, str] = {}
        self._recent: list[bool] = []    # last outcomes, newest last
        self._physics = game_physics(cfg)

    # --- the same small interface as ExperimentPlan -------------------------
    @property
    def total(self) -> int:
        return self.warmup + self.levels * self.per_level

    @property
    def spawned(self) -> int:
        return self._next

    @property
    def exhausted(self) -> bool:
        return self._next >= self.total

    def speed_for(self, index: int) -> float:
        level = max(0, index - self.warmup) // self.per_level
        return self.speed0 + min(level, self.levels - 1) * self.speed_step

    @property
    def current_speed(self) -> float:
        return self.speed_for(max(self._next - 1, 0))

    # --- choosing the next obstacle -----------------------------------------
    def _pick(self, index: int) -> tuple[str, float]:
        if index < self.warmup:
            return "warm", W_MAX
        u = self.rng.random()
        if self._recent[-2:] == [False, False] or u < P_RELIEF:
            return "relief", min(W_MAX, 2.0 * self.w)
        if u < P_RELIEF + P_STRETCH:
            return "stretch", max(W_MIN, 0.7 * self.w)
        return "track", self.w * math.exp(self.rng.uniform(-0.08, 0.08))

    def take_next(self) -> Optional[Trial]:
        if self.exhausted:
            return None
        i = self._next
        kind, w = self._pick(i)
        w = min(W_MAX, max(W_MIN, w))
        self._kinds[i] = kind
        self._next += 1
        speed = self.speed_for(i)
        height = js.height_for_window(w, self.width, speed, self._physics)
        last = self.exhausted
        interval = None if last else self.rng.uniform(*self.interval_range)
        gap = None if last else speed * interval - self.width
        return Trial(index=i, target_w=w, height=height, width=self.width,
                     speed=speed, interval_after=interval, gap_after_px=gap)

    # --- learning from outcomes ---------------------------------------------
    def report(self, index: int, cleared) -> None:
        cleared = bool(cleared)
        self._recent.append(cleared)
        if self._kinds.get(index) == "track":
            step = -STEP_DOWN if cleared else self.step_up
            self.w = min(W_MAX, max(W_MIN, self.w * math.exp(step)))
