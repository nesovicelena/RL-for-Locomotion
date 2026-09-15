"""Terrain curriculum for the ERFI study: one recipe, applied to every (condition, seed).

Each run is trained in stages of increasing heightfield relief, every stage
starting from the previous stage's final parameters (normaliser, actor, critic).
This is the legged_gym / paper-style terrain curriculum, made explicit because
Brax cannot change the MJX model inside one training run.

    python scripts/train_curriculum.py --config configs/experiment/erfi_study_curr.yaml
    python scripts/train_curriculum.py --conditions none rfi --seeds 0 --smoke

Layout, chosen so that scripts/eval.py works unchanged on <out>:

    <out>/<condition>/seed<k>/
        stages/s0_a0.000/   full run directory of stage 0 (see training/ppo.py)
        stages/s1_a0.015/
        ...
        params_final        copied from the last stage
        env_config.json     copied from the last stage (the evaluation terrain)
        ppo_config.json, spec.json
        curve.json          all stages concatenated, steps cumulative, with a "stage" column
        summary.json        condition, seed, stages, total steps and wall time

Finished stages (with params_final) are skipped, so the sweep resumes after a
pod dies. A finished run (params_final at the run root) is skipped entirely.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
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
    p.add_argument("--config", default=str(REPO / "configs/experiment/erfi_study_curr.yaml"))
    p.add_argument("--conditions", nargs="+", choices=list(erfi.CONDITIONS))
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--out", help="run root; relative paths are under RL_EXPERIMENTS_DIR")
    p.add_argument("--impl", choices=["warp", "jax"])
    p.add_argument("--smoke", action="store_true", help="2 stages x 1M steps, 512 envs: pipeline check")
    p.add_argument("--force", action="store_true", help="retrain runs that already finished")
    return p.parse_args()


def stage_dir(run_dir: Path, i: int, amplitude: float) -> Path:
    return run_dir / "stages" / f"s{i}_a{amplitude:.3f}"


def finalize(run_dir: Path, stage_dirs: list[Path], stages: list[dict], t0: float) -> dict:
    """Promote the last stage to the run root and write the combined curve/summary."""
    last = stage_dirs[-1]
    for name in ("params_final", "env_config.json", "ppo_config.json", "spec.json"):
        src = last / name
        if src.is_dir():
            shutil.copytree(src, run_dir / name, dirs_exist_ok=True)
        else:
            shutil.copy2(src, run_dir / name)

    curve, offset = [], 0
    for i, d in enumerate(stage_dirs):
        rows = json.loads((d / "curve.json").read_text())
        for r in rows:
            curve.append({**r, "step": r["step"] + offset, "stage": i, "stage_step": r["step"],
                          "terrain_amplitude": stages[i]["terrain_amplitude"]})
        offset += rows[-1]["step"] if rows else 0
    (run_dir / "curve.json").write_text(json.dumps(curve, indent=2))

    stage_summaries = [json.loads((d / "summary.json").read_text()) for d in stage_dirs]
    summary = {
        "condition": stage_summaries[-1]["condition"],
        "seed": stage_summaries[-1]["seed"],
        "robot": stage_summaries[-1]["robot"],
        "task": stage_summaries[-1]["task"],
        "curriculum": [{"terrain_amplitude": s["terrain_amplitude"], "num_timesteps": s["num_timesteps"]} for s in stages],
        "num_timesteps": sum(s["num_timesteps"] for s in stages),
        "wall_s": sum(s["wall_s"] for s in stage_summaries),
        "final_reward": curve[-1]["reward"] if curve else None,
        "stage_final_rewards": [s["final_reward"] for s in stage_summaries],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    train_cfg = dict(cfg.get("train", {}))
    stages = list(cfg["stages"])
    if args.impl:
        train_cfg["impl"] = args.impl
    if train_cfg.get("task", "rough_terrain") != "rough_terrain":
        sys.exit("the terrain curriculum only makes sense with train.task: rough_terrain")

    conditions = args.conditions or cfg["conditions"]
    seeds = args.seeds or cfg["seeds"]
    out = Path(args.out or cfg.get("out", "erfi_study_curr"))
    if not out.is_absolute():
        out = experiments_root() / out
    if args.smoke:
        stages = [dict(s, num_timesteps=1_000_000) for s in stages[:2]]
        train_cfg["num_evals"] = 2
        train_cfg.setdefault("ppo_overrides", {}).update(num_envs=512, batch_size=64, num_minibatches=8)
        out = out.with_name(out.name + "_smoke")

    print(f"runs -> {out}")
    print(f"robot {train_cfg.get('robot', 'go1')}  conditions {conditions}  seeds {seeds}")
    print("stages: " + ", ".join(f"{s['terrain_amplitude']*100:g} cm x {s['num_timesteps']/1e6:g}M" for s in stages))

    for condition in conditions:
        for seed in seeds:
            run_dir = out / condition / f"seed{seed}"
            if ppo.is_finished(run_dir) and not args.force:
                print(f"skip {run_dir} (finished)")
                continue
            t0 = time.time()
            stage_dirs: list[Path] = []
            prev: Path | None = None
            for i, st in enumerate(stages):
                d = stage_dir(run_dir, i, st["terrain_amplitude"])
                stage_dirs.append(d)
                if ppo.is_finished(d) and not args.force:
                    print(f"  skip stage {i} {d.name} (finished)")
                    prev = d
                    continue
                spec = ppo.TrainSpec(
                    condition=condition, seed=seed, init_from=str(prev) if prev else None,
                    terrain_amplitude=st["terrain_amplitude"], num_timesteps=st["num_timesteps"],
                    **{k: v for k, v in train_cfg.items()},
                )
                print(f"\n=== {condition} seed {seed} stage {i}: amplitude {st['terrain_amplitude']} m, "
                      f"{st['num_timesteps']/1e6:g}M steps -> {d}")
                result = ppo.train(spec, d)
                print(f"=== stage done: reward {result['summary']['final_reward']:.3f}")
                prev = d
            summary = finalize(run_dir, stage_dirs, stages, t0)
            print(f"=== {condition} seed {seed} done: final reward {summary['final_reward']:.3f}, "
                  f"stage rewards {[round(r, 1) for r in summary['stage_final_rewards']]}, "
                  f"{summary['wall_s']/60:.1f} min\n")


if __name__ == "__main__":
    main()
