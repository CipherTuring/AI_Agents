"""Runs every (config, seed) pair in parallel and skips finished runs.

Usage:
    python -m experiments.sweep "experiments/configs/tier1_tabular_*.yaml"
"""
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.train import RESULTS_DIR, apply_smoke, load_config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="config paths or glob patterns")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    paths = sorted(p for pattern in args.configs for p in glob.glob(pattern))
    if not paths:
        raise SystemExit("No config matched.")

    results_dir = RESULTS_DIR / "smoke" if args.smoke else RESULTS_DIR
    jobs = []
    for path in paths:
        config = load_config(path)
        seeds = config["seeds"][:1] if args.smoke else config["seeds"]
        if args.smoke:
            config = apply_smoke(config)
        for seed in seeds:
            done = results_dir / config["name"] / f"seed_{seed}" / "summary.json"
            if done.exists():
                print(f"skip {config['name']} seed {seed}")
            else:
                jobs.append((config, seed))

    print(f"{len(jobs)} runs to do, {args.workers} at a time", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run, config, seed, results_dir): (config["name"], seed)
            for config, seed in jobs
        }
        for future in as_completed(futures):
            name, seed = futures[future]
            s = future.result()
            print(f"done {name} seed {seed}: final {s['final_mean_return']:.1f}, "
                  f"{s['train_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
