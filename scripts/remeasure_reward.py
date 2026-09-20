"""Re-measure every finished run's final reward on a *nominal* environment.

    RL_EXPERIMENTS_DIR=/workspace/experiments/redo python scripts/remeasure_reward.py
    python scripts/remeasure_reward.py --studies erfi_study_v3_rough_l2.5 --conditions dr

Why this exists. Brax applies `randomization_fn` to the evaluation environment
as well as the training one (brax/training/agents/ppo/train.py, the
`_maybe_wrap_env(eval_env or environment, ..., randomization_fn=...)` call), and
the study only passes a randomizer for the `dr` condition. So every number in a
`dr` run's curve.json was measured under randomized dynamics while the other
five conditions were measured on nominal ones: DR ranks last of six on curve
reward in all five Go1 studies and first to third on the perturbation protocol,
which is the signature of a harder exam rather than a worse policy. The
protocol results are unaffected (eval/perturb.py builds its own env and never
sees the randomizer); only the curves are.

This reloads each run's `params_final` and re-runs Brax's own evaluator against
the same env the run trained against, with the randomizer off, so the training
curves become comparable across conditions. It also reports the deterministic
policy, which is what the protocol and any deployment actually use, while
training measured the stochastic one (`deterministic_eval` defaults to False).

Writes per run `nominal_reward.json` and per study `nominal_reward.csv`:

    run, condition, seed, curve_reward, nominal_reward, nominal_reward_std,
    nominal_reward_det, nominal_reward_det_std, episode_length, num_envs, randomized

`curve_reward` is the last point of curve.json, so the bias is visible as
`nominal_reward - curve_reward` on the dr rows and ~0 elsewhere. Evaluated runs
are skipped unless --force, so the script resumes after an interruption.

Cheap: one evaluation epoch per policy, the same cost the training loop paid 15
times per run. Minutes on the pod for the whole tree.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import jax  # noqa: E402

from rl_locomotion.training import ppo  # noqa: E402  (re-exports device_put_replicated for brax)

from brax.io import model as brax_model  # noqa: E402
from brax.training import acting  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from mujoco_playground import wrapper  # noqa: E402

from rl_locomotion.envs import erfi  # noqa: E402

DEFAULT_STUDIES = [
    "erfi_study_l2.5", "erfi_study_rough_l2.5",
    "erfi_study_v3_l2.5", "erfi_study_v3_rough_l2.5",
    "erfi_study_curr_v3_l2.5",
]


def experiments_root() -> Path:
    env = os.environ.get("RL_EXPERIMENTS_DIR")
    if env:
        return Path(env)
    if Path("/workspace").is_dir():
        return Path("/workspace/experiments")
    return REPO / "experiments"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--studies", nargs="+", default=DEFAULT_STUDIES)
    p.add_argument("--conditions", nargs="+", choices=list(erfi.CONDITIONS),
                   help="default: every condition, so one self-consistent set of numbers")
    p.add_argument("--num-envs", type=int, default=128, help="brax's num_eval_envs default")
    p.add_argument("--episode-length", type=int, help="default: the run's own ppo_config value")
    p.add_argument("--impl", choices=["warp", "jax"], help="default: the run's own backend")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default="params_final")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def policy_factory(run_dir: Path, env) -> tuple:
    """(make_policy, params) for a run, rebuilt exactly as training built them."""
    params_cfg = json.loads((run_dir / "ppo_config.json").read_text())
    nf = dict(params_cfg["network_factory"])
    nf["policy_hidden_layer_sizes"] = tuple(nf["policy_hidden_layer_sizes"])
    nf["value_hidden_layer_sizes"] = tuple(nf["value_hidden_layer_sizes"])
    obs_size = env.observation_size
    obs_shape = ({k: (v,) if isinstance(v, int) else tuple(v) for k, v in obs_size.items()}
                 if isinstance(obs_size, dict) else (obs_size,))
    preprocess = running_statistics.normalize if params_cfg.get("normalize_observations", True) else (lambda x, y: x)
    network = ppo_networks.make_ppo_networks(obs_shape, env.action_size, preprocess_observations_fn=preprocess, **nf)
    return ppo_networks.make_inference_fn(network), params_cfg


def measure(run_dir: Path, args: argparse.Namespace) -> dict:
    """Final reward of one run on a nominal env, stochastic and deterministic."""
    summary = json.loads((run_dir / "summary.json").read_text())
    ppo_cfg = json.loads((run_dir / "ppo_config.json").read_text())
    episode_length = args.episode_length or int(ppo_cfg["episode_length"])
    action_repeat = int(ppo_cfg.get("action_repeat", 1))

    # The env training evaluated against: the run's own config with ERFI off
    # (training/ppo.py builds exactly this as `eval_env`). The randomizer, which
    # brax also applied here for `dr`, is simply never passed below.
    cfg = ppo.load_env_config(run_dir)
    if args.impl:
        cfg.impl = args.impl
    cfg.erfi.enable = False
    env = erfi.load(cfg)
    wrapped = wrapper.wrap_for_brax_training(
        env, episode_length=episode_length, action_repeat=action_repeat,
        randomization_fn=None, full_reset=True,
    )

    make_policy, _ = policy_factory(run_dir, env)
    params = brax_model.load_params(str(run_dir / args.checkpoint))

    out = {}
    for label, deterministic in (("", False), ("_det", True)):
        evaluator = acting.Evaluator(
            wrapped, functools.partial(make_policy, deterministic=deterministic),
            num_eval_envs=args.num_envs, episode_length=episode_length,
            action_repeat=action_repeat, key=jax.random.PRNGKey(args.seed),
        )
        metrics = evaluator.run_evaluation(params, training_metrics={})
        out[f"nominal_reward{label}"] = float(metrics["eval/episode_reward"])
        out[f"nominal_reward{label}_std"] = float(metrics["eval/episode_reward_std"])

    curve = json.loads((run_dir / "curve.json").read_text())
    return {
        "condition": summary["condition"],
        "seed": summary["seed"],
        "curve_reward": float(curve[-1]["reward"]) if curve else None,
        **out,
        "episode_length": episode_length,
        "num_envs": args.num_envs,
        # True = this run's curve.json was measured under domain randomization
        # and is therefore not comparable with the other conditions'.
        "randomized": erfi.uses_domain_randomization(summary["condition"]),
    }


def main() -> None:
    args = parse_args()
    root = experiments_root()
    print(f"experiments root {root}\nstudies {args.studies}")

    for study in args.studies:
        study_root = root / study
        if not study_root.is_dir():
            print(f"\n== {study}: not found, skipping")
            continue
        runs = sorted(p.parent for p in study_root.glob(f"*/seed*/{args.checkpoint}"))
        if args.conditions:
            runs = [r for r in runs if r.parent.name in args.conditions]
        if not runs:
            print(f"\n== {study}: no runs to measure")
            continue
        print(f"\n== {study}: {len(runs)} runs")

        rows = []
        for run_dir in runs:
            run_id = str(run_dir.relative_to(study_root))
            dest = run_dir / "nominal_reward.json"
            if dest.exists() and not args.force:
                rows.append({"run": run_id, **json.loads(dest.read_text())})
                print(f"  skip {run_id} (measured)")
                continue
            row = measure(run_dir, args)
            dest.write_text(json.dumps(row, indent=2))
            rows.append({"run": run_id, **row})
            delta = row["nominal_reward"] - row["curve_reward"] if row["curve_reward"] is not None else float("nan")
            print(f"  {run_id:16s} curve {row['curve_reward']:7.3f} -> nominal {row['nominal_reward']:7.3f} "
                  f"({delta:+6.3f}){'  <- was randomized' if row['randomized'] else ''}"
                  f"   deterministic {row['nominal_reward_det']:7.3f}", flush=True)

        df = pd.DataFrame(rows)
        df.to_csv(study_root / "nominal_reward.csv", index=False)
        print(f"  -> {study_root / 'nominal_reward.csv'}")
        bias = df[df.randomized].nominal_reward.mean() - df[df.randomized].curve_reward.mean() if df.randomized.any() else None
        if bias is not None:
            print(f"  dr curve bias: {bias:+.2f} reward (nominal minus curve)")


if __name__ == "__main__":
    main()
