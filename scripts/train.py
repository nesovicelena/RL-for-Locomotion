"""Train the Go1 ERFI study: every (condition, seed) pair, sequentially.

Usage (on the pod, inside tmux so a dropped SSH connection does not kill it):

    python scripts/train.py --config configs/experiment/erfi_study.yaml
    python scripts/train.py --conditions none erfi_50 --seeds 0 --smoke   # pipeline check
    python scripts/train.py --conditions rao --seeds 1 --num-timesteps 50_000_000

Finished runs (those with a params_final) are skipped, so re-running the same
command resumes the sweep.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rl_locomotion.envs import erfi  # noqa: E402
from rl_locomotion.training import ppo  # noqa: E402


def experiments_root() -> Path:
    env = os.environ.get("RL_EXPERIMENTS_DIR")
    if env:
        return Path(env)
    if Path("/workspace").is_dir():
        return Path("/workspace/experiments")
    return REPO / "experiments"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO / "configs/experiment/erfi_study.yaml"))
    p.add_argument("--conditions", nargs="+", choices=list(erfi.CONDITIONS))
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--out", help="run root; relative paths are under RL_EXPERIMENTS_DIR")
    p.add_argument("--num-timesteps", type=int)
    p.add_argument("--num-evals", type=int)
    p.add_argument("--history-len", type=int)
    p.add_argument("--policy-layers", type=int, nargs="+")
    p.add_argument("--value-layers", type=int, nargs="+")
    p.add_argument("--symmetric-critic", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--rfi-lim", type=float)
    p.add_argument("--rao-lim", type=float)
    p.add_argument("--impl", choices=["warp", "jax"])
    p.add_argument("--smoke", action="store_true",
                   help="2M steps, 2 evals, 512 envs: checks the pipeline end to end")
    p.add_argument("--force", action="store_true", help="retrain runs that already finished")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    train_cfg = dict(cfg.get("train", {}))

    for key in ("num_timesteps", "num_evals", "history_len", "policy_layers", "value_layers",
                "symmetric_critic", "rfi_lim", "rao_lim", "impl"):
        val = getattr(args, key)
        if val is not None:
            train_cfg[key] = val

    conditions = args.conditions or cfg["conditions"]
    seeds = args.seeds or cfg["seeds"]
    out = Path(args.out or cfg.get("out", "erfi_study"))
    if not out.is_absolute():
        out = experiments_root() / out
    if args.smoke:
        train_cfg.update(num_timesteps=2_000_000, num_evals=2)
        train_cfg.setdefault("ppo_overrides", {}).update(num_envs=512, batch_size=64, num_minibatches=8)
        out = out.with_name(out.name + "_smoke")

    print(f"runs -> {out}")
    print(f"conditions {conditions}  seeds {seeds}")
    for condition in conditions:
        for seed in seeds:
            run_dir = out / condition / f"seed{seed}"
            if ppo.is_finished(run_dir) and not args.force:
                print(f"skip {run_dir} (finished)")
                continue
            spec = ppo.TrainSpec(condition=condition, seed=seed, **train_cfg)
            print(f"\n=== {condition} seed {seed} -> {run_dir}")
            result = ppo.train(spec, run_dir)
            s = result["summary"]
            print(f"=== done: reward {s['final_reward']:.3f} in {s['wall_s'] / 60:.1f} min\n")


if __name__ == "__main__":
    main()
