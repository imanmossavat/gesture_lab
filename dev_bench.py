"""Headless development bench for the adaptive difficulty system.

This script does not open a window, a camera, or MediaPipe.  It exercises the
pure-Python logic in ``adaptive.py`` and produces tables / plots for tuning.

Examples:
    python dev_bench.py feasibility
    python dev_bench.py sweep
    python dev_bench.py simulate --frames 3000 --latency 60
    python dev_bench.py latency-report --output latency.png
    python dev_bench.py compare --output compare.png
    python dev_bench.py report
"""
import argparse
import csv
import dataclasses
import datetime
import math
import random
import statistics
import subprocess
import sys

from adaptive import (
    AdaptiveController,
    Config,
    OUTCOME_HEADER,
    PRESETS,
    air_time,
    apex_time,
    clears_obstacle,
    difficulty_profile,
    jump_peak,
    timing_window,
)


def _closed_unit(name: str):
    def f(value):
        v = float(value)
        if not 0.0 <= v <= 1.0:
            raise argparse.ArgumentTypeError(f"{name} must be in [0, 1], got {v}")
        return v
    return f


def _open_unit(name: str):
    def f(value):
        v = float(value)
        if not 0.0 < v < 1.0:
            raise argparse.ArgumentTypeError(f"{name} must be in (0, 1), got {v}")
        return v
    return f


def _half_open_unit(name: str):
    def f(value):
        v = float(value)
        if not 0.0 <= v < 1.0:
            raise argparse.ArgumentTypeError(f"{name} must be in [0, 1), got {v}")
        return v
    return f


def _positive_float_upto_one(name: str):
    def f(value):
        v = float(value)
        if not 0.0 < v <= 1.0:
            raise argparse.ArgumentTypeError(f"{name} must be in (0, 1], got {v}")
        return v
    return f


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def _write_csv_meta(file, config, extra=None):
    """Write config metadata as CSV comment lines before the header."""
    meta = dict(extra or {})
    meta["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["git_hash"] = _git_hash()
    meta["command"] = " ".join(sys.argv)
    for key, value in dataclasses.asdict(config).items():
        meta.setdefault(key, value)
    for key, value in meta.items():
        file.write(f"# {key}={value}\n")


def _simulate(
    config: Config,
    rng: random.Random,
    frames: int,
    timing_noise_ms: float,
    latency_jitter_ms: float,
    miss_prob: float,
    eta: float | None = None,
    target: float | None = None,
    start_d: float | None = None,
    momentum: float | None = None,
    seed: int = 42,
) -> tuple[AdaptiveController, list[int]]:
    """Run a deterministic player simulation and return the controller + per-obstacle rows."""
    ctrl_rng = random.Random(seed)
    cfg_kwargs: dict[str, float] = {}
    if eta is not None:
        cfg_kwargs["ADAPTATION_RATE"] = eta
    if target is not None:
        cfg_kwargs["TARGET_SUCCESS_RATE"] = target
    if start_d is not None:
        cfg_kwargs["D_START"] = start_d
    if momentum is not None:
        cfg_kwargs["MOMENTUM"] = momentum
    if cfg_kwargs:
        controller_cfg = Config(**cfg_kwargs)
    else:
        controller_cfg = config
    controller = AdaptiveController(controller_cfg, rng=ctrl_rng)

    rows = []
    for frame in range(frames):
        d_spawn = controller.d
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
        rows.append({
            "t_s": frame,
            "player": "dev_bench",
            "session": 1,
            "mode": 0,
            "d_spawn": f"{d_spawn:.4f}",
            "d_after": f"{controller.d:.4f}",
            "speed": f"{speed:.1f}",
            "height": f"{h:.1f}",
            "width": f"{w:.1f}",
            "outcome": 1 if success else 0,
        })
    return controller, rows


def cmd_feasibility(args: argparse.Namespace) -> int:
    cfg = Config()
    air = air_time(cfg)
    print(f"Feasibility check (gap mode: {args.gap_mode})")
    print(
        f"{'d':>4} {'speed':>6} {'h':>6} {'w':>6} "
        f"{'gap_px':>7} {'slack_s':>8} {'status':>10}"
    )
    rc = 0
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        profile = difficulty_profile(d, cfg)
        # Worst-case height with +10% jitter.
        h = profile.height * 1.1
        if args.gap_mode == "schedule":
            jitter_min = cfg.SPAWN_INTERVAL_JITTER[0]
            gap_px = profile.speed * profile.spawn_interval * jitter_min - profile.width
        else:  # fixed650
            gap_px = 650.0
        slack_s = gap_px / profile.speed - air
        margin = timing_window(h, profile.width, profile.speed, cfg)
        if slack_s < cfg.LANDING_MARGIN_S:
            status = "tight"
            rc = 1
        elif margin <= 0.0:
            status = "impossible"
            rc = 1
        else:
            status = "ok"
        print(
            f"{d:4.2f} {profile.speed:6.1f} {h:6.1f} {profile.width:6.1f} "
            f"{gap_px:7.1f} {slack_s:8.4f} {status:>10}"
        )
    return rc


def cmd_report(args: argparse.Namespace) -> int:
    """Print regenerable design tables derived entirely from Config."""
    cfg = Config()
    air = air_time(cfg)
    apex = apex_time(cfg)
    peak = jump_peak(cfg)

    print("Design report (all values derived from adaptive.Config)")
    print(
        f"Operating range: d in [{cfg.D_MIN}, {cfg.D_MAX}]  "
        f"target success rate = {cfg.TARGET_SUCCESS_RATE}"
    )
    print()
    print("Physics")
    print(f"  air time       = {air:.3f} s")
    print(f"  apex time      = {apex:.3f} s")
    print(f"  jump peak      = {peak:.1f} px")
    print()

    ds = [cfg.D_MIN] + sorted(p["d"] for p in PRESETS.values()) + [cfg.D_MAX]
    print("Difficulty profile")
    print(f"{'d':>5} {'speed':>7} {'height':>7} {'width':>7} {'interval':>9}")
    for d in ds:
        prof = difficulty_profile(d, cfg)
        print(
            f"{d:5.2f} {prof.speed:7.1f} {prof.height:7.1f} {prof.width:7.1f} "
            f"{prof.spawn_interval:9.3f}"
        )
    print()

    print("Minimum gap (worst case: land before next obstacle)")
    print(f"{'d':>5} {'speed':>7} {'required_gap_px':>15} {'status':>10}")
    for d in ds:
        prof = difficulty_profile(d, cfg)
        required = max(cfg.MIN_GAP_FLOOR_PX, prof.speed * (air + cfg.LANDING_MARGIN_S))
        status = "ok" if required >= cfg.MIN_GAP_FLOOR_PX else "floor"
        print(f"{d:5.2f} {prof.speed:7.1f} {required:15.1f} {status:>10}")
    print()

    print("Clearance budget (height +10% jitter)")
    print(f"{'d':>5} {'speed':>7} {'window_s':>10} {'window_ms':>10} {'half_ms':>10}")
    budgets_ms: list[tuple[float, float]] = []
    for d in ds:
        prof = difficulty_profile(d, cfg)
        h = prof.height * (1.0 + cfg.HEIGHT_JITTER)
        window = timing_window(h, prof.width, prof.speed, cfg)
        half_ms = 0.5 * window * 1000.0
        budgets_ms.append((d, half_ms))
        print(
            f"{d:5.2f} {prof.speed:7.1f} {window:10.4f} "
            f"{window * 1000.0:10.2f} {half_ms:10.2f}"
        )
    print()

    # Operating range: timing-noise std σ for which an interior equilibrium exists.
    # Model: reaction error ε ~ N(0, σ²), success iff |ε| < half-window.
    target = cfg.TARGET_SUCCESS_RATE
    z = statistics.NormalDist().inv_cdf((1.0 + target) / 2.0)  # P(|Z| < z) = target
    sigmas = [(d, half / z) for d, half in budgets_ms]
    print(
        f"Interior equilibrium (Gaussian reaction error, P(clear) = {target:.2f})"
    )
    print(f"{'d':>5} {'half_ms':>10} {'sigma_ms':>10}")
    for d, sigma in sigmas:
        half = next(half for dd, half in budgets_ms if dd == d)
        print(f"{d:5.2f} {half:10.2f} {sigma:10.1f}")
    sigma_min, sigma_max = min(s for _, s in sigmas), max(s for _, s in sigmas)
    print(f"  σ range for interior d ∈ [{cfg.D_MIN}, {cfg.D_MAX}]: "
          f"{sigma_min:.1f}–{sigma_max:.1f} ms")
    print()

    print("Controller update per outcome")
    success_delta = cfg.ADAPTATION_RATE * (1.0 - cfg.TARGET_SUCCESS_RATE)
    failure_delta = cfg.ADAPTATION_RATE * (0.0 - cfg.TARGET_SUCCESS_RATE)
    print(f"  success delta  = +{success_delta:.4f} d")
    print(f"  failure delta  = {failure_delta:.4f} d")
    if cfg.MOMENTUM > 0.0:
        long_run = cfg.ADAPTATION_RATE / (1.0 - cfg.MOMENTUM)
        print(f"  long-run gain  = {long_run:.4f} d/outcome (momentum {cfg.MOMENTUM})")
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
    controller, rows = _simulate(
        cfg,
        rng,
        frames=args.frames,
        timing_noise_ms=args.noise,
        latency_jitter_ms=args.latency,
        miss_prob=args.miss,
        seed=args.seed,
        start_d=args.start_d,
        eta=args.adaptation_rate,
        target=args.target_success_rate,
        momentum=args.momentum,
    )
    outcomes = [int(r["outcome"]) for r in rows]
    overall = sum(outcomes) / len(outcomes)
    tail_n = min(args.tail, len(outcomes))
    tail_rate = sum(outcomes[-tail_n:]) / tail_n
    print(f"Simulation: seed={args.seed} frames={args.frames}")
    print(f"  noise={args.noise:.1f}ms latency={args.latency:.1f}ms miss={args.miss:.2f}")
    print(f"  final d = {controller.d:.3f}")
    print(f"  overall success = {overall:.3f}")
    print(f"  last {tail_n} success  = {tail_rate:.3f}")
    print(f"  outcomes: {controller.outcomes}  ok={controller.successes}  bad={controller.failures}")
    if args.csv:
        controller_cfg = controller.config
        with open(args.csv, "w", newline="") as f:
            _write_csv_meta(
                f, controller_cfg,
                extra={
                    "log_type": "bench_simulate",
                    "seed": args.seed,
                    "frames": args.frames,
                    "noise_ms": args.noise,
                    "latency_jitter_ms": args.latency,
                    "miss_prob": args.miss,
                    "tail": args.tail,
                },
            )
            writer = csv.DictWriter(f, fieldnames=OUTCOME_HEADER)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV written to {args.csv}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    cfg = Config()
    print("Comparing controller variants")
    variants = [
        ("slow", 0.04, cfg.TARGET_SUCCESS_RATE, None, None),
        ("default", cfg.ADAPTATION_RATE, cfg.TARGET_SUCCESS_RATE, None, None),
        ("fast", 0.12, cfg.TARGET_SUCCESS_RATE, None, None),
        ("target=0.80", cfg.ADAPTATION_RATE, 0.80, None, None),
        # lower eta to offset the ~1/(1-momentum) gain of a steady streak
        ("mom=0.5", cfg.ADAPTATION_RATE / 2, cfg.TARGET_SUCCESS_RATE, None, 0.5),
        ("mom=0.8", cfg.ADAPTATION_RATE / 5, cfg.TARGET_SUCCESS_RATE, None, 0.8),
    ]
    if (
        args.start_d is not None
        or args.adaptation_rate is not None
        or args.target_success_rate is not None
        or args.momentum is not None
    ):
        variants.append((
            "custom",
            args.adaptation_rate if args.adaptation_rate is not None else cfg.ADAPTATION_RATE,
            args.target_success_rate if args.target_success_rate is not None else cfg.TARGET_SUCCESS_RATE,
            args.start_d,
            args.momentum,
        ))
    print(f"{'variant':>12} {'final d':>8} {'overall':>8} {'last 400':>10}")
    summary_rows = []
    all_rows = []
    for name, eta, target, start_d, momentum in variants:
        rng = random.Random(args.seed)
        controller, variant_rows = _simulate(
            cfg, rng, args.frames, args.noise, args.latency, args.miss,
            eta=eta, target=target, start_d=start_d, momentum=momentum,
            seed=args.seed,
        )
        outcomes = [int(r["outcome"]) for r in variant_rows]
        overall = sum(outcomes) / len(outcomes)
        tail_rate = sum(outcomes[-400:]) / 400
        print(f"{name:>12} {controller.d:8.3f} {overall:8.3f} {tail_rate:10.3f}")
        summary_rows.append((name, overall, tail_rate))
        for r in variant_rows:
            r["variant"] = name
            all_rows.append(r)

    if args.output:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            names = [r[0] for r in summary_rows]
            overalls = [r[1] for r in summary_rows]
            tails = [r[2] for r in summary_rows]
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
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            _write_csv_meta(f, cfg, extra={"log_type": "bench_compare", "seed": args.seed})
            writer = csv.DictWriter(f, fieldnames=OUTCOME_HEADER[:4] + ["variant"] + OUTCOME_HEADER[4:])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"CSV written to {args.csv}")
    return 0


def cmd_latency_report(args: argparse.Namespace) -> int:
    cfg = Config()
    latencies = list(range(0, args.max_latency + 1, args.step))
    print("Latency impact (fixed timing noise = 80 ms, miss = 5%)")
    print(f"{'latency_ms':>10} {'final d':>8} {'overall':>8} {'last 400':>10}")
    summary_rows = []
    all_rows = []
    for lat in latencies:
        rng = random.Random(args.seed)
        controller, lat_rows = _simulate(
            cfg, rng, args.frames, args.noise, lat, args.miss,
            seed=args.seed,
            start_d=args.start_d,
            eta=args.adaptation_rate,
            target=args.target_success_rate,
            momentum=args.momentum,
        )
        outcomes = [int(r["outcome"]) for r in lat_rows]
        overall = sum(outcomes) / len(outcomes)
        tail_rate = sum(outcomes[-400:]) / 400
        print(f"{lat:>10} {controller.d:8.3f} {overall:8.3f} {tail_rate:10.3f}")
        summary_rows.append((lat, overall, tail_rate, controller.d))
        for r in lat_rows:
            r["latency_ms"] = lat
            all_rows.append(r)

    if args.output:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = [r[0] for r in summary_rows]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
            ax1.plot(xs, [r[1] for r in summary_rows], label="overall")
            ax1.plot(xs, [r[2] for r in summary_rows], label="last 400")
            ax1.axhline(cfg.TARGET_SUCCESS_RATE, color="red", linestyle="--")
            ax1.set_ylabel("success rate")
            ax1.set_ylim(0, 1)
            ax1.legend()
            ax2.plot(xs, [r[3] for r in summary_rows], color="green")
            ax2.set_xlabel("end-to-end latency jitter (ms)")
            ax2.set_ylabel("final difficulty d")
            ax2.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(args.output)
            print(f"Saved plot to {args.output}")
        except Exception as exc:
            print(f"Could not save plot: {exc}", file=sys.stderr)
    if args.csv:
        controller_cfg = controller.config
        with open(args.csv, "w", newline="") as f:
            _write_csv_meta(
                f, controller_cfg,
                extra={
                    "log_type": "bench_latency_report",
                    "seed": args.seed,
                    "max_latency": args.max_latency,
                    "step": args.step,
                    "noise_ms": args.noise,
                    "miss_prob": args.miss,
                },
            )
            writer = csv.DictWriter(f, fieldnames=OUTCOME_HEADER[:4] + ["latency_ms"] + OUTCOME_HEADER[4:])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"CSV written to {args.csv}")
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
    p_feas.add_argument(
        "--gap-mode",
        choices=["fixed650", "schedule"],
        default="schedule",
        help="how to measure the gap between obstacles (default: schedule)",
    )
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
    p_sim.add_argument("--start-d", type=_closed_unit("--start-d"), default=None, help="initial d (0-1)")
    p_sim.add_argument("--adaptation-rate", type=_positive_float_upto_one("--adaptation-rate"), default=None, help="controller learning rate (0-1]")
    p_sim.add_argument("--target-success-rate", type=_open_unit("--target-success-rate"), default=None, help="target success rate (0-1)")
    p_sim.add_argument("--momentum", type=_half_open_unit("--momentum"), default=None, help="controller momentum, independent of the learning rate [0, 1)")
    p_sim.add_argument("--csv", type=str, default=None, help="write per-obstacle rows to CSV")
    p_sim.set_defaults(func=cmd_simulate)

    p_comp = sub.add_parser("compare", help="compare controller tuning variants")
    p_comp.add_argument("--frames", type=int, default=2000)
    p_comp.add_argument("--noise", type=float, default=80.0)
    p_comp.add_argument("--latency", type=float, default=90.0)
    p_comp.add_argument("--miss", type=float, default=0.05)
    p_comp.add_argument("--seed", type=int, default=42)
    p_comp.add_argument("--output", type=str, default=None)
    p_comp.add_argument("--start-d", type=_closed_unit("--start-d"), default=None, help="initial d for the custom variant (0-1)")
    p_comp.add_argument("--adaptation-rate", type=_positive_float_upto_one("--adaptation-rate"), default=None, help="learning rate for the custom variant (0-1]")
    p_comp.add_argument("--target-success-rate", type=_open_unit("--target-success-rate"), default=None, help="target success rate for the custom variant (0-1)")
    p_comp.add_argument("--momentum", type=_half_open_unit("--momentum"), default=None, help="controller momentum, independent of the learning rate [0, 1)")
    p_comp.add_argument("--csv", type=str, default=None, help="write per-obstacle rows to CSV")
    p_comp.set_defaults(func=cmd_compare)

    p_lat = sub.add_parser("latency-report", help="sweep latency and report success rates")
    p_lat.add_argument("--frames", type=int, default=1500)
    p_lat.add_argument("--noise", type=float, default=80.0)
    p_lat.add_argument("--miss", type=float, default=0.05)
    p_lat.add_argument("--max-latency", type=int, default=200)
    p_lat.add_argument("--step", type=int, default=20)
    p_lat.add_argument("--seed", type=int, default=42)
    p_lat.add_argument("--output", type=str, default=None)
    p_lat.add_argument("--start-d", type=_closed_unit("--start-d"), default=None, help="initial d (0-1)")
    p_lat.add_argument("--adaptation-rate", type=_positive_float_upto_one("--adaptation-rate"), default=None, help="controller learning rate (0-1]")
    p_lat.add_argument("--target-success-rate", type=_open_unit("--target-success-rate"), default=None, help="target success rate (0-1)")
    p_lat.add_argument("--momentum", type=_half_open_unit("--momentum"), default=None, help="controller momentum, independent of the learning rate [0, 1)")
    p_lat.add_argument("--csv", type=str, default=None, help="write per-obstacle rows to CSV")
    p_lat.set_defaults(func=cmd_latency_report)

    p_play = sub.add_parser("play", help="note about interactive play")
    p_play.set_defaults(func=cmd_play)

    p_rep = sub.add_parser("report", help="print regenerable design tables")
    p_rep.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
