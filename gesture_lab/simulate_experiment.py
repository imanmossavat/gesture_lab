#!/usr/bin/env python3
"""Run the experiment with a simulated player and write a trial log.

No camera, no window.  Useful for three things:

1. See what the analysis output looks like before you have real data.
2. Check that the pipeline works end to end: if the simulated player's jumps
   are timed inside the physics window, the game should say "cleared".
3. Test an analysis on a player whose true behaviour you know (you choose the
   bias and spread), so you can see whether the analysis recovers it.

Example:
    python -m gesture_lab.simulate_experiment --mu 0.02 --sigma 0.05 --out sim_trials.csv
    python -m gesture_lab.analyze_trials sim_trials.csv --plot sim.png

The simulated player knows the window and aims at its centre with a random
error:   t_j = centre + mu + N(0, sigma^2).
The loop below follows the same order of steps as game.py.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys

from gesture_lab.adaptive import (
    AdaptiveController, Config, attempt_jump, resolve_obstacles,
)
from gesture_lab import experiment as ex

DT = 1.0 / 60.0  # the game runs at 60 frames per second


def simulate(
    mu: float = 0.0,
    sigma: float = 0.05,
    seed: int = 0,
    player: str = "sim",
    speed: float = 400.0,
    blocks: int = 4,
    cfg: Config | None = None,
) -> list[dict]:
    """Play one session and return the trial rows (dicts keyed by TRIAL_HEADER)."""
    cfg = cfg or Config()
    rng = random.Random(seed)

    plan = ex.ExperimentPlan(ex.make_trials(cfg, rng, speed=speed, blocks=blocks))
    recorder = ex.TrialRecorder(cfg)
    controller = AdaptiveController(cfg, rng=rng)   # required by resolve_obstacles

    rows: list[dict] = []
    obstacles: list = []
    # What this simulated player plans for each obstacle: the jump time it will
    # aim for, and whether it has already jumped.  (A list of small records,
    # not a dict keyed by id(): Python can reuse the id of a deleted object.)
    plans: list[dict] = []

    player_front = cfg.PLAYER_X + cfg.PLAYER_WIDTH
    player_y = cfg.GROUND_Y - cfg.PLAYER_HEIGHT
    velocity_y = 0.0
    play_time = 0.0
    spawn_timer = 0.8
    pending_interval = None

    def on_outcome(ob, outcome):
        row = recorder.finish(ob, outcome, play_time, player=player, session=1)
        if row is not None:
            rows.append(row)

    while True:
        # 1. The player decides to jump: aim at the window centre plus an error.
        plans = [pl for pl in plans if pl["ob"] in obstacles and pl["ob"].x > player_front - 1]
        for pl in plans:
            ob = pl["ob"]
            if pl["jumped"] or ob.x <= player_front:
                continue
            time_to_front = (ob.x - player_front) / speed
            if time_to_front <= -pl["target"]:       # the aimed-for moment has come
                velocity_y, accepted = attempt_jump(
                    player_y, velocity_y, cfg.GROUND_Y,
                    cfg.PLAYER_HEIGHT, cfg.JUMP_VELOCITY,
                )
                if accepted:
                    recorder.note_jump(speed)
                    pl["jumped"] = True
            break  # only the nearest obstacle matters to the player

        # 2. Player physics (same as game.py).
        velocity_y += cfg.GRAVITY * DT
        player_y += velocity_y * DT
        if player_y > cfg.GROUND_Y - cfg.PLAYER_HEIGHT:
            player_y = cfg.GROUND_Y - cfg.PLAYER_HEIGHT

        # 3. Spawn the next planned obstacle when its timer runs out.
        spawn_timer -= DT
        if spawn_timer <= 0.0 and not plan.exhausted:
            trial = plan.take_next()
            ob = ex.trial_to_obstacle(trial, cfg)
            obstacles.append(ob)
            recorder.register(
                ob, target_w=trial.target_w,
                next_gap_px=trial.gap_after_px, index=trial.index,
            )
            window = ex.game_window(ob.h, ob.w, speed, trial.gap_after_px, cfg)
            plans.append({"ob": ob, "jumped": False,
                          "target": window.centre + mu + rng.gauss(0.0, sigma)})
            spawn_timer = trial.interval_after or 0.0

        # 4. Move obstacles and decide outcomes (this is where the game scores).
        obstacles = resolve_obstacles(
            obstacles, cfg.PLAYER_X, player_y, cfg.PLAYER_WIDTH,
            cfg.PLAYER_HEIGHT, cfg.GROUND_Y, speed, DT, controller, cfg,
            on_outcome=on_outcome,
        )
        play_time += DT

        if plan.exhausted and all(ob.done for ob in obstacles):
            return rows


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        f.write("# log_type=trials\n# source=simulate_experiment.py\n")
        writer = csv.writer(f)
        writer.writerow(ex.TRIAL_HEADER)
        for row in rows:
            writer.writerow(ex.row_values(row))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mu", type=float, default=0.0, help="bias in seconds (+ = late)")
    p.add_argument("--sigma", type=float, default=0.05, help="timing spread in seconds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blocks", type=int, default=4, help="blocks of 7 obstacles (default 4)")
    p.add_argument("--out", type=str, default="sim_trials.csv")
    args = p.parse_args(argv)

    rows = simulate(args.mu, args.sigma, args.seed, blocks=args.blocks)
    write_csv(rows, args.out)
    cleared = sum(int(r["outcome"]) for r in rows)
    print(f"Wrote {len(rows)} obstacles to {args.out} ({cleared} cleared)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
