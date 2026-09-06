"""Evaluate every finished run of the ERFI study under the perturbation protocol.

    python scripts/eval.py --config configs/experiment/erfi_study.yaml --plot
    python scripts/eval.py --runs /workspace/experiments/erfi_study --params payload_kg push_N

Writes <runs>/results.csv (one row per run x parameter x level), a per-condition
summary table, and with --plot the Fig.-5-style success curves as PNG.
Already-evaluated runs are skipped unless --force is given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rl_locomotion.envs import erfi  # noqa: E402
from rl_locomotion.eval import perturb  # noqa: E402
from rl_locomotion.training import ppo  # noqa: E402


def experiments_root() -> Path:
    env = os.environ.get("RL_EXPERIMENTS_DIR")
    if env:
        return Path(env)
    if Path("/workspace").is_dir():
        return Path("/workspace/experiments")
    return REPO / "experiments"


def discover_runs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/seed*/params_final"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO / "configs/experiment/erfi_study.yaml"))
    p.add_argument("--runs", help="run root (default: <out> from the config)")
    p.add_argument("--params", nargs="+", choices=list(perturb.PROTOCOL))
    p.add_argument("--n-episodes", type=int)
    p.add_argument("--impl", choices=["warp", "jax"])
    p.add_argument("--checkpoint", default="params_final")
    p.add_argument("--seed", type=int, default=0, help="seed for the evaluation episodes")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    ev = dict(cfg.get("eval", {}))
    impl = args.impl or ev.pop("impl", "jax")
    if args.n_episodes:
        ev["n_episodes"] = args.n_episodes
    if args.params:
        ev["params"] = args.params
    ev["command"] = tuple(ev.get("command", (0.5, 0.0, 0.0)))
    ev["params"] = tuple(ev.get("params", perturb.PROTOCOL))
    spec = perturb.EvalSpec(**ev)

    root = Path(args.runs or cfg.get("out", "erfi_study"))
    if not root.is_absolute():
        root = experiments_root() / root
    runs = discover_runs(root)
    if not runs:
        sys.exit(f"no finished runs under {root}")

    results_path = root / "results.csv"
    existing = pd.read_csv(results_path) if results_path.exists() and not args.force else pd.DataFrame()
    frames = [existing] if len(existing) else []
    done_runs = set(existing["run"]) if len(existing) else set()

    for run_dir in runs:
        run_id = str(run_dir.relative_to(root))
        if run_id in done_runs:
            print(f"skip {run_id} (evaluated)")
            continue
        summary = json.loads((run_dir / "summary.json").read_text())
        print(f"\n=== {run_id}")
        env = erfi.load(perturb.eval_env_config(ppo.load_env_config(run_dir), impl=impl))
        policy = ppo.load_policy(run_dir, env, checkpoint=args.checkpoint)
        df = perturb.evaluate_policy(
            env, policy, spec, seed=args.seed,
            meta={"run": run_id, "condition": summary["condition"], "seed": summary["seed"]},
        )
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(results_path, index=False)

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(results_path, index=False)
    print(f"\nresults -> {results_path}")

    table = perturb.summary_table(results)
    table.to_csv(root / "summary_success_rate.csv")
    print("\nmean success rate over all levels:\n", table.to_string())

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        for metric in ("success_rate", "fall_rate", "progress_m"):
            fig = perturb.plot_success_curves(results, metric=metric)
            path = root / f"{metric}_curves.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"figure -> {path}")


if __name__ == "__main__":
    main()
