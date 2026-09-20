"""Measure a trained policy's per-joint actuator torque and derive the ERFI limit vector.

The Go1 study set one torque limit for every joint (2.5 Nm, about 10 % of the
motor limit and half a stance torque). The humanoid's joints differ by an order
of magnitude in the torque they carry, so the limit is a vector: the paper's
"half a stance torque" applied per joint (docs/humanoid_design.md 4.1).

    python scripts/measure_stance_torque.py /workspace/experiments/redo/erfi_study_bh_rough/none/seed0
    python scripts/measure_stance_torque.py <run_dir> --fraction 0.5 \
        --write configs/experiment/erfi_study_bh_rough.yaml configs/experiment/erfi_study_bh.yaml

Rolls the policy through the protocol's nominal setting (ERFI off, 0.5 m/s,
8 s, 50 episodes, no perturbation) on the JAX backend, records
`data.actuator_force` at every control step the robot is still up, and reports
the per-joint RMS. The limit is `fraction` x RMS, averaged over the left and
right leg so the vector is symmetric. Writes stance_torque.json next to the run
and, with --write, replaces the `rfi_lim:` / `rao_lim:` lines of the given
experiment configs in place (comments and everything else untouched).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rl_locomotion.envs import bh, erfi  # noqa: E402
from rl_locomotion.envs import spot as spot_pkg  # noqa: E402
from rl_locomotion.eval import perturb  # noqa: E402
from rl_locomotion.training import ppo  # noqa: E402


def measure(env, policy, n_episodes: int = 50, duration_s: float = 8.0,
            command=(0.5, 0.0, 0.0), seed: int = 0) -> dict:
    """Per-joint RMS of actuator_force over the alive steps of `n_episodes` nominal episodes."""
    n_steps = int(round(duration_s / env.dt))
    cmd = jp.array(command)
    nu = env.mjx_model.nu

    def episode(key):
        key, k_reset = jax.random.split(key)
        state = env.reset(k_reset)
        state.info["command"] = cmd

        def body(carry, _):
            state, key, fallen, sq_sum, n_alive, progress0 = carry
            key, k_act = jax.random.split(key)
            action, _ = policy(state.obs, k_act)
            state = env.step(state, action)
            state.info["command"] = cmd
            fallen = fallen | (state.done > 0)
            f = state.data.actuator_force
            sq_sum = jp.where(fallen, sq_sum, sq_sum + f * f)
            n_alive = jp.where(fallen, n_alive, n_alive + 1)
            return (state, key, fallen, sq_sum, n_alive, progress0), None

        init = (state, key, jp.zeros((), bool), jp.zeros(nu), jp.zeros((), jp.int32),
                state.data.xpos[env._torso_body_id][:2])
        (state, _, fallen, sq_sum, n_alive, pos0), _ = jax.lax.scan(body, init, None, length=n_steps)
        progress = jp.linalg.norm(state.data.xpos[env._torso_body_id][:2] - pos0)
        return sq_sum, n_alive, fallen, progress

    keys = jax.random.split(jax.random.PRNGKey(seed), n_episodes)
    sq_sum, n_alive, fallen, progress = jax.device_get(jax.jit(jax.vmap(episode))(keys))
    rms = np.sqrt(sq_sum.sum(0) / max(int(n_alive.sum()), 1))
    return {
        "rms_per_joint": rms.tolist(),
        "fall_rate": float(np.mean(fallen)),
        "mean_progress_m": float(np.mean(progress)),
        "alive_steps": int(n_alive.sum()),
        "n_episodes": n_episodes,
        "duration_s": duration_s,
        "command": list(command),
    }


def symmetric_limit(rms: np.ndarray, fraction: float) -> list[float]:
    """fraction x RMS, averaged between the two legs (first and second half of the vector)."""
    half = len(rms) // 2
    per_leg = 0.5 * (rms[:half] + rms[half:])
    lim = np.round(fraction * per_leg, 3)
    return np.concatenate([lim, lim]).tolist()


def write_limits(config_path: Path, limits: list[float], provenance: str) -> None:
    text = config_path.read_text()
    value = "[" + ", ".join(f"{x:.3f}" for x in limits) + "]"
    for key in ("rfi_lim", "rao_lim"):
        pattern = re.compile(rf"^(\s*){key}:.*$", re.MULTILINE)
        if not pattern.search(text):
            raise ValueError(f"{config_path} has no `{key}:` line to replace")
        text = pattern.sub(rf"\g<1>{key}: {value}   # {provenance}", text, count=1)
    config_path.write_text(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="run directory of a trained (walking) policy, usually the best `none` seed")
    p.add_argument("--checkpoint", default="params_final")
    p.add_argument("--fraction", type=float, default=0.5, help="limit = fraction x RMS (paper: half a stance torque)")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--duration-s", type=float, default=8.0)
    p.add_argument("--impl", default="jax")
    p.add_argument("--write", nargs="*", default=[], help="experiment YAMLs whose rfi_lim/rao_lim lines to replace")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    env = erfi.load(perturb.eval_env_config(ppo.load_env_config(run_dir), impl=args.impl))
    policy = ppo.load_policy(run_dir, env, checkpoint=args.checkpoint)
    result = measure(env, policy, n_episodes=args.n_episodes, duration_s=args.duration_s)
    rms = np.asarray(result["rms_per_joint"])
    limits = symmetric_limit(rms, args.fraction)
    names = {"bh": bh.JOINT_NAMES, "spot": spot_pkg.JOINT_NAMES}.get(env.robot, [f"j{i}" for i in range(len(rms))])

    result.update({
        "run_dir": str(run_dir), "checkpoint": args.checkpoint, "robot": env.robot, "task": env.task,
        "fraction": args.fraction, "limit": limits, "joint_names": list(names), "date": date.today().isoformat(),
    })
    (run_dir / "stance_torque.json").write_text(json.dumps(result, indent=2))

    print(f"policy {run_dir}  fall rate {result['fall_rate']:.2f}  mean progress {result['mean_progress_m']:.2f} m")
    if result["mean_progress_m"] < 2.5:
        print("WARNING: this policy does not reach 2.5 m at nominal; measure a walking policy instead")
    print(f"{'joint':8s} {'RMS Nm':>8s} {'limit Nm':>9s}")
    for n, r, l in zip(names, rms, limits):
        print(f"{n:8s} {r:8.3f} {l:9.3f}")

    provenance = f"measured {result['date']}: {args.fraction} x RMS actuator force of {run_dir.name} ({run_dir.parent.name}), {args.n_episodes} x {args.duration_s:g} s"
    for cfg in args.write:
        write_limits(Path(cfg), limits, provenance)
        print(f"wrote limits -> {cfg}")


if __name__ == "__main__":
    main()
