"""Render a trained ERFI policy walking, as an MP4.

    # the best policy of a study (highest mean success over the whole protocol)
    python scripts/make_video.py --study experiments/redo/erfi_study_rough_l2.5
    # a specific run, custom command and camera
    python scripts/make_video.py --run experiments/redo/erfi_study_l2.5/erfi_50/seed1 \
        --command 0.8 0 0 --seconds 6 --camera side
    # a rough-trained policy on flat ground, or on a harsher heightfield
    python scripts/make_video.py --run ... --task flat_terrain
    python scripts/make_video.py --run ... --terrain-amplitude 0.10

The policy is deterministic and ERFI is off, as at deployment. The command is
held fixed for the whole clip (default 0.5 m/s forward, the protocol command).
Output goes to experiments/videos/<study>_<condition>_seed<k>[_<task>].mp4 unless
--out is given. Runs on the laptop: a 10 s clip takes about a minute to render.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rl_locomotion.eval.render import configure_backend  # noqa: E402

configure_backend()  # MUJOCO_GL before MuJoCo is imported (glfw on macOS, egl on the pod)

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import mediapy as media  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rl_locomotion.envs import erfi  # noqa: E402
from rl_locomotion.eval import perturb  # noqa: E402
from rl_locomotion.training import ppo  # noqa: E402


def best_run(study: Path) -> Path:
    """The walking run with the highest mean success rate over all protocol levels."""
    r = pd.read_csv(study / "results.csv")
    nominal = r[(r.param == "payload_kg") & (r.level == 0.0)].set_index("run")
    walking = nominal[nominal.progress_m >= 1.0].index
    score = r[r.run.isin(walking)].groupby("run").success_rate.mean().sort_values(ascending=False)
    print("top runs by mean success:\n" + score.head(5).round(3).to_string())
    return study / score.index[0]


def parse_schedule(entries: list[str]) -> list[tuple[float, tuple[float, float, float]]]:
    """`["5:0.5,0,0", "5:0.5,0,0.8"]` -> `[(5.0, (0.5, 0, 0)), (5.0, (0.5, 0, 0.8))]`."""
    segments = []
    for entry in entries:
        secs, sep, command = entry.partition(":")
        values = [float(v) for v in command.split(",")] if sep else []
        if len(values) != 3:
            raise SystemExit(f"bad schedule segment {entry!r}; expected SEC:VX,VY,WZ (e.g. 5:0.5,0,0.8)")
        segments.append((float(secs), (values[0], values[1], values[2])))
    return segments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=Path, help="run directory with params_final")
    src.add_argument("--study", type=Path, help="study directory with results.csv; picks its best run")
    p.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--schedule", nargs="+", metavar="SEC:VX,VY,WZ",
                   help="command schedule instead of one fixed command; overrides --command and "
                        "--seconds. E.g. turn left then right: "
                        "--schedule 5:0.5,0,0 5:0.5,0,0.8 5:0.5,0,-0.8")
    p.add_argument("--camera", default="track", help="track | side | top | back (Go1/A1 scene cameras)")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--task", choices=list(erfi.TASKS), help="override the training terrain")
    p.add_argument("--terrain-amplitude", type=float, help="override heightfield relief (m), rough terrain only")
    p.add_argument("--terrain-shape", choices=("playground", "bowl", "rough_bowl"),
                   help="override terrain shape, rough terrain only")
    p.add_argument("--slope-deg", type=float, help="bowl slope (deg), with --terrain-shape bowl / rough_bowl")
    p.add_argument("--seed", type=int, default=0, help="reset seed (spawn pose and velocity)")
    p.add_argument("--checkpoint", default="params_final")
    p.add_argument("--out", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run or best_run(args.study)
    overrides = {}
    if args.task:
        overrides["task"] = args.task
    if args.terrain_amplitude is not None:
        overrides["terrain_amplitude"] = args.terrain_amplitude
    if args.terrain_shape:
        overrides["terrain_shape"] = args.terrain_shape
    if args.slope_deg is not None:
        overrides["slope_deg"] = args.slope_deg
    cfg = perturb.eval_env_config(ppo.load_env_config(run, **overrides), impl="jax")
    env = erfi.load(cfg)
    policy = ppo.load_policy(run, env, checkpoint=args.checkpoint)
    # One fixed command, or a schedule of (duration, command) segments.
    segments = parse_schedule(args.schedule) if args.schedule else [(args.seconds, tuple(args.command))]
    total_s = sum(secs for secs, _ in segments)
    print(f"run {run}\nrobot {env.robot}  task {env.task}  relief {env.terrain_amplitude:.3f} m  "
          f"{total_s:g} s")
    for secs, c in segments:
        print(f"  {secs:5.1f} s  command {c}")

    n_steps = int(round(total_s / env.dt))
    # Per-step command, so a segment boundary lands on a control step.
    per_step: list[tuple[float, float, float]] = []
    for secs, c in segments:
        per_step += [c] * int(round(secs / env.dt))
    per_step = (per_step + [segments[-1][1]] * n_steps)[:n_steps]
    commands = jp.array(per_step, dtype=jp.float32)

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.PRNGKey(args.seed))
    state.info["command"] = commands[0]
    key = jax.random.PRNGKey(1)
    states, fell = [state], False
    for t in range(n_steps):
        key, k = jax.random.split(key)
        action, _ = policy(state.obs, k)
        state = step(state, action)
        state.info["command"] = commands[t]  # hold it; the env would resample it
        states.append(state)
        if float(state.done) > 0 and not fell:
            fell = True
            print(f"robot fell at t = {(t + 1) * env.dt:.2f} s")
    x0, x1 = np.array(states[0].data.xpos[env._torso_body_id]), np.array(states[-1].data.xpos[env._torso_body_id])
    print(f"travelled {np.linalg.norm((x1 - x0)[:2]):.2f} m in {n_steps * env.dt:.1f} s")

    # Playground's rough scene defines no headlight and renders almost black;
    # give every scene the same lighting as the flat one.
    vis = env.mj_model.vis
    vis.headlight.active = 1
    vis.headlight.ambient[:] = 0.3
    vis.headlight.diffuse[:] = 0.8
    vis.headlight.specular[:] = 0.5
    frames = env.render(states, height=args.height, width=args.width, camera=args.camera)
    fps = 1.0 / env.dt
    if args.out is None:
        study = run.parent.parent.name
        suffix = f"_{args.task}" if args.task else ""
        if args.terrain_amplitude is not None:
            suffix += f"_a{args.terrain_amplitude:.3f}"
        if args.terrain_shape:
            suffix += f"_{args.terrain_shape}"
        if args.slope_deg is not None:
            suffix += f"_{args.slope_deg:g}deg"
        if args.schedule:
            suffix += "_schedule"
        args.out = REPO / "experiments" / "videos" / f"{study}_{run.parent.name}_{run.name}{suffix}.mp4"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(args.out), frames, fps=fps)
    print(f"video -> {args.out}  ({len(frames)} frames, {fps:.0f} fps)")


if __name__ == "__main__":
    main()
