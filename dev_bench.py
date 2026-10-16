"""Headless development bench for the adaptive difficulty system.

This script does not open a window, a camera, or MediaPipe.  It exercises the
pure-Python logic in ``adaptive.py`` and produces tables / plots for tuning.

Examples:
    python dev_bench.py feasibility
    python dev_bench.py sweep
    python dev_bench.py simulate --frames 3000 --latency 60
    python dev_bench.py latency-report --output latency.png
    python dev_bench.py compare --output compare.png
"""
import argparse
import math
import random
import sys

from adaptive import (
    AdaptiveController,
    Config,
    clears_obstacle,
    difficulty_profile,
    timing_window,
)


def _simulate(
    config: Config,
    rng: random.Random,
    frames: int,
    timing_noise_ms: float,
    latency_jitter_ms: float,
    miss_prob: float,
    eta: float | None = None,
    target: float | None = None,
    seed: int = 42,
) -> tuple[AdaptiveController, list[int]]:
    """Run a deterministic player simulation and return the controller + outcomes."""
    ctrl_rng = random.Random(seed)
    if eta is not None or target is not None:
        cfg = Config() if eta is None else Config(ADAPTATION_RATE=eta)
        if target is not None:
            cfg = Config(
                ADAPTATION_RATE=cfg.ADAPTATION_RATE,
                TARGET_SUCCESS_RATE=target,
            )
        controller = AdaptiveController(cfg, rng=ctrl_rng)
    else:
        controller = AdaptiveController(config, rng=ctrl_rng)

    outcomes = []
    for _ in range(frames):
        profile = difficulty_profile(controller.d, config)
        h = profile.height * rng.uniform(0.9, 1.1)
        w = profile.width
        speed = profile.speed
        if rng.random() < miss_prob:
            success = False
        else:
            sigma = math.hypot(timing_noise_ms, latency_jitter_ms)
            offset_ms = rng.gauss(0.0, sigma)
            success = clears_obstacle(h, w, speed, offset_ms / 1000.0, config)
        controller.update(1 if success else 0)
        outcomes.append(1 if success else 0)
    return controller, outcomes


def cmd_feasibility(args: argparse.Namespace) -> int:
    cfg = Config()
    print("Feasibility check (theoretical clearance margin in seconds)")
    print(f"{'d':>4} {'speed':>6} {'h':>6} {'w':>6} {'margin':>10} {'status':>10}")
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        profile = difficulty_profile(d, cfg)
        # Worst-case height with +10% jitter.
        h = profile.height * 1.1
        margin = timing_window(h, profile.width, profile.speed, cfg)
        status = "feasible" if margin > 0 else "impossible"
        print(
            f"{d:4.2f} {profile.speed:6.1f} {h:6.1f} {profile.width:6.1f} "
            f"{margin:10.4f} {status:>10}"
        )
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    cfg = Config()
    print("Difficulty sweep")
    print(f"{'d':>4} {'speed':>6} {'h':>6} {'w':>6} {'interval':>9} {'margin@0':>10}")
    d = 0.0
    while d <= 1.0001:
        profile = difficulty_profile(d, cfg)
        h = profile.height * 1.1
        margin = timing_window(h, profile.width, profile.speed, cfg)
        print(
            f"{d:4.2f} {profile.speed:6.1f} {h:6.1f} {profile.width:6.1f} "
            f"{profile.spawn_interval:9.3f} {margin:10.4f}"
        )
        d += args.step
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    cfg = Config()
    rng = random.Random(args.seed)
    controller, outcomes = _simulate(
        cfg,
        rng,
        frames=args.frames,
        timing_noise_ms=args.noise,
        latency_jitter_ms=args.latency,
        miss_prob=args.miss,
        seed=args.seed,
    )
    overall = sum(outcomes) / len(outcomes)
    tail_n = min(args.tail, len(outcomes))
    tail_rate = sum(outcomes[-tail_n:]) / tail_n
    print(f"Simulation: seed={args.seed} frames={args.frames}")
    print(f"  noise={args.noise:.1f}ms latency={args.latency:.1f}ms miss={args.miss:.2f}")
    print(f"  final d = {controller.d:.3f}")
    print(f"  overall success = {overall:.3f}")
    print(f"  last {tail_n} success  = {tail_rate:.3f}")
    print(f"  outcomes: {controller.outcomes}  ok={controller.successes}  bad={controller.failures}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    cfg = Config()
    print("Comparing adaptation rates")
    variants = [
        ("slow", 0.04, cfg.TARGET_SUCCESS_RATE),
        ("default", cfg.ADAPTATION_RATE, cfg.TARGET_SUCCESS_RATE),
        ("fast", 0.12, cfg.TARGET_SUCCESS_RATE),
        ("target=0.80", cfg.ADAPTATION_RATE, 0.80),
    ]
    print(f"{'variant':>12} {'final d':>8} {'overall':>8} {'last 400':>10}")
    rows = []
    for name, eta, target in variants:
        rng = random.Random(args.seed)
        controller, outcomes = _simulate(
            cfg, rng, args.frames, args.noise, args.latency, args.miss,
            eta=eta, target=target, seed=args.seed,
        )
        overall = sum(outcomes) / len(outcomes)
        tail_rate = sum(outcomes[-400:]) / 400
        print(f"{name:>12} {controller.d:8.3f} {overall:8.3f} {tail_rate:10.3f}")
        rows.append((name, overall, tail_rate))

    if args.output:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            names = [r[0] for r in rows]
            overalls = [r[1] for r in rows]
            tails = [r[2] for r in rows]
            x = range(len(names))
            width = 0.35
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar([i - width / 2 for i in x], overalls, width, label="overall")
            ax.bar([i + width / 2 for i in x], tails, width, label="last 400")
            ax.axhline(cfg.TARGET_SUCCESS_RATE, color="red", linestyle="--", label="target")
            ax.set_ylabel("success rate")
            ax.set_title("Controller variants")
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.set_ylim(0, 1)
            ax.legend()
            plt.tight_layout()
            plt.savefig(args.output)
            print(f"Saved plot to {args.output}")
        except Exception as exc:
            print(f"Could not save plot: {exc}", file=sys.stderr)
    return 0


def cmd_latency_report(args: argparse.Namespace) -> int:
    cfg = Config()
    latencies = list(range(0, args.max_latency + 1, args.step))
    print("Latency impact (fixed timing noise = 80 ms, miss = 5%)")
    print(f"{'latency_ms':>10} {'final d':>8} {'overall':>8} {'last 400':>10}")
    rows = []
    for lat in latencies:
        rng = random.Random(args.seed)
        controller, outcomes = _simulate(
            cfg, rng, args.frames, args.noise, lat, args.miss, seed=args.seed,
        )
        overall = sum(outcomes) / len(outcomes)
        tail_rate = sum(outcomes[-400:]) / 400
        print(f"{lat:>10} {controller.d:8.3f} {overall:8.3f} {tail_rate:10.3f}")
        rows.append((lat, overall, tail_rate, controller.d))

    if args.output:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = [r[0] for r in rows]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
            ax1.plot(xs, [r[1] for r in rows], label="overall")
            ax1.plot(xs, [r[2] for r in rows], label="last 400")
            ax1.axhline(cfg.TARGET_SUCCESS_RATE, color="red", linestyle="--")
            ax1.set_ylabel("success rate")
            ax1.set_ylim(0, 1)
            ax1.legend()
            ax2.plot(xs, [r[3] for r in rows], color="green")
            ax2.set_xlabel("end-to-end latency jitter (ms)")
            ax2.set_ylabel("final difficulty d")
            ax2.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(args.output)
            print(f"Saved plot to {args.output}")
        except Exception as exc:
            print(f"Could not save plot: {exc}", file=sys.stderr)
    return 0


def cmd_play(_args: argparse.Namespace) -> int:
    print(
        "Interactive play is intentionally not implemented in dev_bench.\n"
        "Run the game instead:\n\n"
        "  python game.py\n\n"
        "For a headless smoke test:\n\n"
        "  SDL_VIDEODRIVER=dummy python game.py --dummy-camera --smoke-frames 120\n"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless adaptive-difficulty bench")
    sub = parser.add_subparsers(dest="command", required=True)

    p_feas = sub.add_parser("feasibility", help="check theoretical clearance at key difficulties")
    p_feas.set_defaults(func=cmd_feasibility)

    p_sweep = sub.add_parser("sweep", help="sweep d and print parameters")
    p_sweep.add_argument("--step", type=float, default=0.05)
    p_sweep.set_defaults(func=cmd_sweep)

    p_sim = sub.add_parser("simulate", help="run a seeded player simulation")
    p_sim.add_argument("--frames", type=int, default=2000)
    p_sim.add_argument("--noise", type=float, default=80.0)
    p_sim.add_argument("--latency", type=float, default=90.0)
    p_sim.add_argument("--miss", type=float, default=0.05)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--tail", type=int, default=400)
    p_sim.set_defaults(func=cmd_simulate)

    p_comp = sub.add_parser("compare", help="compare controller tuning variants")
    p_comp.add_argument("--frames", type=int, default=2000)
    p_comp.add_argument("--noise", type=float, default=80.0)
    p_comp.add_argument("--latency", type=float, default=90.0)
    p_comp.add_argument("--miss", type=float, default=0.05)
    p_comp.add_argument("--seed", type=int, default=42)
    p_comp.add_argument("--output", type=str, default=None)
    p_comp.set_defaults(func=cmd_compare)

    p_lat = sub.add_parser("latency-report", help="sweep latency and report success rates")
    p_lat.add_argument("--frames", type=int, default=1500)
    p_lat.add_argument("--noise", type=float, default=80.0)
    p_lat.add_argument("--miss", type=float, default=0.05)
    p_lat.add_argument("--max-latency", type=int, default=200)
    p_lat.add_argument("--step", type=int, default=20)
    p_lat.add_argument("--seed", type=int, default=42)
    p_lat.add_argument("--output", type=str, default=None)
    p_lat.set_defaults(func=cmd_latency_report)

    p_play = sub.add_parser("play", help="note about interactive play")
    p_play.set_defaults(func=cmd_play)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
