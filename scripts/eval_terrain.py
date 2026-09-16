"""Terrain evaluation suites: every finished policy of one or more studies, on terrains
it may never have seen.

    # all four suites, on the five Go1 studies of the redo tree
    RL_EXPERIMENTS_DIR=/workspace/experiments/redo python scripts/eval_terrain.py
    # a subset
    python scripts/eval_terrain.py --studies erfi_study_v3_rough_l2.5 --suites bowl_slope rough_relief

Suites (all on the rough_terrain scene, so a policy trained on flat ground is
rebuilt on the heightfield scene; the observation is identical):

    bowl_slope           smooth bowl, uphill slope 0 / 10 / 20 / 30 deg
    rough_bowl_slope     bowl with the 5 cm rocky relief on top, same slopes
    rough_relief         Playground's rocky field at 5 / 7 / 8 / 9 / 10 cm relief
    rough_bowl_protocol  the paper's five sweeps (payload, push, friction, gravity, Kp)
                         on a 10 deg rough bowl

The success criterion is the protocol's: 0.5 m/s forward for 8 s, no fall and
>= 2.5 m along the initial heading. Each suite writes <study>/results_<suite>.csv,
summary_success_rate_<suite>.csv and the three curve PNGs; rows carry `task`,
`terrain_shape`, `trained_task` and `nominal`. Evaluated runs are skipped, so the
script resumes after an interruption. Run on the pod (GPU); ~1 h per study.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rl_locomotion.envs import erfi  # noqa: E402
from rl_locomotion.eval import perturb  # noqa: E402
from rl_locomotion.training import ppo  # noqa: E402

DEFAULT_STUDIES = [
    "erfi_study_l2.5", "erfi_study_rough_l2.5",          # v1 flat / rough
    "erfi_study_v3_l2.5", "erfi_study_v3_rough_l2.5",    # v3 flat / rough
    "erfi_study_curr_v3_l2.5",                           # terrain curriculum, v3 observation
]

ROUGH_FRICTION = [0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3]  # centred on the rough scene's 1.0

# suite -> env overrides (on top of the run's config) and protocol spec pieces
SUITES: dict[str, dict] = {
    "bowl_slope": dict(
        env=dict(task="rough_terrain", terrain_shape="bowl", slope_deg=0.0),
        params=("slope_deg",), levels={"slope_deg": [0.0, 10.0, 20.0, 30.0]},
    ),
    "rough_bowl_slope": dict(
        env=dict(task="rough_terrain", terrain_shape="rough_bowl", slope_deg=0.0, terrain_amplitude=0.05),
        params=("slope_deg",), levels={"slope_deg": [0.0, 10.0, 20.0, 30.0]},
    ),
    "rough_relief": dict(
        env=dict(task="rough_terrain", terrain_shape="playground", terrain_amplitude=0.05),
        params=("terrain_amplitude",), levels={"terrain_amplitude": [0.05, 0.07, 0.08, 0.09, 0.10]},
    ),
    "rough_bowl_protocol": dict(
        env=dict(task="rough_terrain", terrain_shape="rough_bowl", slope_deg=10.0, terrain_amplitude=0.05),
        params=perturb.STANDARD_PARAMS, levels={"friction": ROUGH_FRICTION},
    ),
}


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
    p.add_argument("--studies", nargs="+", default=DEFAULT_STUDIES, help="study directories under RL_EXPERIMENTS_DIR")
    p.add_argument("--suites", nargs="+", default=list(SUITES), choices=list(SUITES))
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--impl", choices=["warp", "jax"], default="jax")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default="params_final")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def run_suite(study_root: Path, suite: str, args: argparse.Namespace) -> None:
    cfg = SUITES[suite]
    runs = discover_runs(study_root)
    if not runs:
        print(f"  no finished runs under {study_root}, skipping")
        return
    results_path = study_root / f"results_{suite}.csv"
    existing = pd.read_csv(results_path) if results_path.exists() and not args.force else pd.DataFrame()
    frames = [existing] if len(existing) else []
    done = set(existing["run"]) if len(existing) else set()
    spec = perturb.EvalSpec(n_episodes=args.n_episodes, params=tuple(cfg["params"]), levels=dict(cfg["levels"]))

    for run_dir in runs:
        run_id = str(run_dir.relative_to(study_root))
        if run_id in done:
            print(f"  skip {run_id} (evaluated)")
            continue
        summary = json.loads((run_dir / "summary.json").read_text())
        print(f"\n  === {study_root.name} / {run_id}  [{suite}]", flush=True)
        env = erfi.load(perturb.eval_env_config(ppo.load_env_config(run_dir, **cfg["env"]), impl=args.impl))
        policy = ppo.load_policy(run_dir, env, checkpoint=args.checkpoint)
        df = perturb.evaluate_policy(
            env, policy, spec, seed=args.seed,
            meta={"run": run_id, "condition": summary["condition"], "seed": summary["seed"],
                  "study": study_root.name, "suite": suite,
                  "trained_task": summary.get("task", "flat_terrain"),
                  "terrain_shape": cfg["env"].get("terrain_shape", "playground"),
                  "base_slope_deg": cfg["env"].get("slope_deg", 0.0),
                  "base_amplitude": cfg["env"].get("terrain_amplitude", 0.05)},
        )
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(results_path, index=False)

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(results_path, index=False)
    perturb.summary_table(results).to_csv(study_root / f"summary_success_rate_{suite}.csv")
    print(f"\n  results -> {results_path}")
    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        for metric in ("success_rate", "fall_rate", "progress_m"):
            fig = perturb.plot_success_curves(results, metric=metric)
            fig.savefig(study_root / f"{metric}_curves_{suite}.png", dpi=150, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    root = experiments_root()
    print(f"experiments root {root}\nstudies {args.studies}\nsuites {args.suites}")
    for study in args.studies:
        study_root = root / study
        if not study_root.is_dir():
            print(f"\n== {study}: not found, skipping")
            continue
        for suite in args.suites:
            print(f"\n== {study} :: {suite}")
            run_suite(study_root, suite, args)


if __name__ == "__main__":
    main()
