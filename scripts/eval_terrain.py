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

The levels above are the quadruped ones. The robot is read from each study's own
runs and the suite is adjusted for it (`ROBOT_ADJUST`): the Berkeley Humanoid
gets gentler slopes, the two named push axes instead of one random direction,
payload and push as fractions of its own mass, kneeling counted as a fall, and
the keyframe-at-rest start. Go1 and A1 behave exactly as before.

The success criterion is the protocol's: 0.5 m/s forward for 8 s, no fall and
>= 2.5 m along the initial heading. Each suite writes <study>/results_<suite>.csv,
summary_success_rate_<suite>.csv and the three curve PNGs; rows carry `task`,
`terrain_shape`, `trained_task`, `robot` and `nominal`. Evaluated runs are skipped,
so the script resumes after an interruption. Run on the pod (GPU); ~1 h per study.
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
    "erfi_study_l2.5", "erfi_study_rough_l2.5",          # Go1 v1 flat / rough
    "erfi_study_v3_l2.5", "erfi_study_v3_rough_l2.5",    # Go1 v3 flat / rough
    "erfi_study_curr_v3_l2.5",                           # Go1 terrain curriculum, v3 observation
    "erfi_study_bh", "erfi_study_bh_rough",              # Berkeley Humanoid flat / rough
    "erfi_study_curr_bh",                                # Berkeley Humanoid terrain curriculum
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

# Per-robot adjustments to a suite, applied by `spec_for`. Robots absent from
# this table (go1, a1) use the levels above unchanged.
ROBOT_ADJUST: dict[str, dict] = {
    "bh": dict(
        # Kneeling is not a termination on a feet-only-collision humanoid, and the
        # paper deploys from the same pose every time (docs/humanoid_design.md 2.3, 8.1).
        spec=dict(low_base_fraction=0.5, nominal_reset=True),
        # A biped is far weaker laterally than fore-aft, and the completed studies
        # show the two axes ordering the conditions in opposite directions, so they
        # are never averaged into one random-direction sweep.
        split_push=True,
        levels={
            # Measured on erfi_study_bh_rough/erfi_50/seed0 (8 s, the protocol's
            # own criterion): success is 1.0 at 0 and 5 deg, 0.0 from 10 deg up,
            # so the grid resolves 2.5-7.5 deg where the cliff is and keeps two
            # levels past it. Go1's 0/10/20/30 would show only the cliff.
            "slope_deg": [0.0, 2.5, 5.0, 7.5, 10.0, 15.0],
            # Same policy, same criterion: 1.0 at 0.05 m, 0.67 at 0.075, 0.33 at
            # 0.10, 0.0 from 0.125 up. Feet are 0.08 x 0.028 m boxes, so relief
            # bites sooner than on Go1, whose suite runs 0.05 to 0.10.
            "terrain_amplitude": [0.05, 0.075, 0.10, 0.125, 0.15],
        },
        # Fractions of the robot's own mass (payload) and weight m g (pushes),
        # resolved against the model at evaluation time. The completed studies
        # saturate to zero success above 0.20 of weight, so the grid stops at 0.24.
        level_fractions={
            "payload_kg": [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20],
            "push_N_sagittal": [0.0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.20, 0.24],
            "push_N_lateral": [0.0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.20, 0.24],
        },
    ),
}


def study_robot(runs: list[Path]) -> str:
    """The robot a study's runs were trained on, from the first run's env_config.json."""
    return str(ppo.load_env_config(runs[0]).get("robot", "go1"))


def spec_for(suite: str, robot: str, n_episodes: int) -> perturb.EvalSpec:
    """The suite's protocol spec, adjusted for the robot (see ROBOT_ADJUST)."""
    cfg = SUITES[suite]
    params = list(cfg["params"])
    levels = dict(cfg["levels"])
    adjust = ROBOT_ADJUST.get(robot, {})

    if adjust.get("split_push") and "push_N" in params:
        i = params.index("push_N")
        params[i : i + 1] = ["push_N_sagittal", "push_N_lateral"]
    for key, value in adjust.get("levels", {}).items():
        if key in params:
            levels[key] = list(value)
    fractions = {k: list(v) for k, v in adjust.get("level_fractions", {}).items() if k in params}
    # An absolute level always wins over a fraction; drop the ones we replaced.
    levels = {k: v for k, v in levels.items() if k not in fractions}

    return perturb.EvalSpec(
        n_episodes=n_episodes, params=tuple(params), levels=levels,
        level_fractions=fractions, **adjust.get("spec", {}),
    )


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
    robot = study_robot(runs)
    spec = spec_for(suite, robot, args.n_episodes)
    print(f"  robot {robot}  params {spec.params}")

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
