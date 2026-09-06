"""Robustness protocol of Campanaro et al. (arXiv:2209.12878, Sec. VII), on Go1.

The paper deploys each policy with a fixed 0.5 m/s forward command for 8 s and
calls the attempt a success if the robot neither falls nor fails to advance
2.5 m. One simulation parameter is altered at a time. The ANYmal C ranges are
rescaled here to Go1's 12.7 kg (ANYmal C: ~50 kg, 27 kg base):

    payload_kg   extra mass on the trunk           paper: base 22-65 kg (nominal 27)
    push_N       horizontal force on the trunk, 3 s paper: 0-150 N
    friction     floor friction coefficient         paper: 0.2-0.8 (training 0.5; ours 0.6)
    gravity      m/s^2                              paper: -18 .. -2
    kp_scale     PD position gain multiplier        paper: hardware test at Kp/3

ERFI is switched off at evaluation, as in the paper (deployment uses the plain
PD controller). Episodes are batched with vmap and stepped with lax.scan, and
the perturbed MJX model is a traced argument, so one compile per policy covers
every parameter level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jp
import numpy as np
import pandas as pd

from rl_locomotion.envs import erfi

PolicyFn = Callable[[Any, jax.Array], tuple[jax.Array, Any]]

# parameter -> (levels, value during training)
PROTOCOL: dict[str, tuple[list[float], float]] = {
    "payload_kg": ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.0),
    "push_N": ([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0], 0.0),
    "friction": ([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 0.6),
    "gravity": ([-2.0, -5.0, -7.0, -9.81, -12.0, -15.0, -18.0], -9.81),
    "kp_scale": ([0.33, 0.5, 0.75, 1.0, 1.25, 1.5], 1.0),
}

LABELS = {
    "payload_kg": "payload on trunk (kg)",
    "push_N": "push force on trunk, 3 s (N)",
    "friction": "floor friction coefficient",
    "gravity": "gravity (m/s²)",
    "kp_scale": "Kp multiplier",
}


@dataclass
class EvalSpec:
    command: tuple[float, float, float] = (0.5, 0.0, 0.0)
    duration_s: float = 8.0
    success_distance_m: float = 2.5
    n_episodes: int = 50
    push_start_s: float = 1.0
    push_duration_s: float = 3.0
    params: tuple[str, ...] = tuple(PROTOCOL)
    levels: dict[str, list[float]] = field(default_factory=dict)  # override PROTOCOL levels

    def levels_for(self, param: str) -> list[float]:
        return self.levels.get(param, PROTOCOL[param][0])


def eval_env_config(train_env_cfg: Any, impl: str = "jax") -> Any:
    """Training env config -> evaluation env config (ERFI off, no built-in kicks)."""
    cfg = erfi.default_config() if train_env_cfg is None else train_env_cfg
    cfg = cfg.copy_and_resolve_references()
    cfg.erfi.enable = False
    cfg.pert_config.enable = False
    cfg.impl = impl
    return cfg


def perturbed_model(env: Any, param: str, level: float) -> Any:
    """A copy of the env's MJX model with one parameter altered."""
    model = env.mjx_model
    torso = env._torso_body_id
    if param == "payload_kg":
        m0 = model.body_mass[torso]
        scale = (m0 + level) / m0
        return model.tree_replace({
            "body_mass": model.body_mass.at[torso].set(m0 + level),
            "body_inertia": model.body_inertia.at[torso].set(model.body_inertia[torso] * scale),
        })
    if param == "friction":
        fid = env._floor_geom_id
        return model.tree_replace({"geom_friction": model.geom_friction.at[fid, 0].set(level)})
    if param == "gravity":
        return model.replace(opt=model.opt.replace(gravity=jp.array([0.0, 0.0, level])))
    if param == "kp_scale":
        return model.tree_replace({
            "actuator_gainprm": model.actuator_gainprm.at[:, 0].multiply(level),
            "actuator_biasprm": model.actuator_biasprm.at[:, 1].multiply(level),
        })
    if param == "push_N":
        return model  # applied as an external force during the rollout
    raise ValueError(f"unknown perturbation {param!r}")


def make_batched_rollout(env: Any, policy: PolicyFn, spec: EvalSpec):
    """Compile `run(model, keys, push_force) -> per-episode results` for this env+policy."""
    n_steps = int(round(spec.duration_s / env.dt))
    push_start = int(round(spec.push_start_s / env.dt))
    push_end = push_start + int(round(spec.push_duration_s / env.dt))
    command = jp.array(spec.command)
    torso = env._torso_body_id
    nbody = env.mjx_model.nbody

    def episode(key: jax.Array, push_force: jax.Array) -> dict[str, jax.Array]:
        key, k_reset, k_dir = jax.random.split(key, 3)
        state = env.reset(k_reset)
        state.info["command"] = command
        angle = jax.random.uniform(k_dir, minval=0.0, maxval=2 * jp.pi)
        push_dir = jp.array([jp.cos(angle), jp.sin(angle), 0.0])

        # Heading = body x-axis projected on the ground; progress is measured along it.
        x_axis = state.data.xmat[torso][:, 0]
        heading = x_axis[:2] / (jp.linalg.norm(x_axis[:2]) + 1e-6)
        pos0 = state.data.xpos[torso][:2]

        def body(carry, t):
            state, key, fallen, progress, err_sum = carry
            key, k_act = jax.random.split(key)
            action, _ = policy(state.obs, k_act)

            active = (t >= push_start) & (t < push_end)
            xfrc = jp.zeros((nbody, 6)).at[torso, :3].set(push_force * push_dir * active)
            state = state.replace(data=state.data.replace(xfrc_applied=xfrc))

            state = env.step(state, action)
            state.info["command"] = command

            now_fallen = fallen | (state.done > 0)
            cur = jp.dot(state.data.xpos[torso][:2] - pos0, heading)
            progress = jp.where(now_fallen, progress, cur)  # freeze at the fall
            v = env.get_local_linvel(state.data)
            err = jp.sum(jp.square(command[:2] - v[:2]))
            err_sum = jp.where(now_fallen, err_sum, err_sum + err)
            return (state, key, now_fallen, progress, err_sum), None

        init = (state, key, jp.zeros((), bool), jp.zeros(()), jp.zeros(()))
        (_, _, fallen, progress, err_sum), _ = jax.lax.scan(body, init, jp.arange(n_steps))
        success = (~fallen) & (progress >= spec.success_distance_m)
        return {
            "success": success,
            "fallen": fallen,
            "progress_m": progress,
            "tracking_rmse": jp.sqrt(err_sum / n_steps),
        }

    def run(model: Any, keys: jax.Array, push_force: jax.Array) -> dict[str, jax.Array]:
        # Swap the traced model in for the duration of tracing only. Cached
        # executions never re-enter this Python body.
        saved = env._mjx_model
        env._mjx_model = model
        try:
            return jax.vmap(episode, in_axes=(0, None))(keys, push_force)
        finally:
            env._mjx_model = saved

    return jax.jit(run), n_steps


def evaluate_policy(
    env: Any,
    policy: PolicyFn,
    spec: EvalSpec | None = None,
    seed: int = 0,
    meta: dict[str, Any] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the whole protocol for one policy. One row per (param, level)."""
    spec = spec or EvalSpec()
    run, _ = make_batched_rollout(env, policy, spec)
    keys = jax.random.split(jax.random.PRNGKey(seed), spec.n_episodes)

    rows = []
    for param in spec.params:
        for level in spec.levels_for(param):
            model = perturbed_model(env, param, level)
            push = jp.asarray(level if param == "push_N" else 0.0, dtype=jp.float32)
            out = jax.device_get(run(model, keys, push))
            row = {
                **(meta or {}),
                "param": param,
                "level": level,
                "success_rate": float(np.mean(out["success"])),
                "fall_rate": float(np.mean(out["fallen"])),
                "progress_m": float(np.mean(out["progress_m"])),
                "tracking_rmse": float(np.mean(out["tracking_rmse"])),
                "n_episodes": spec.n_episodes,
            }
            rows.append(row)
            if verbose:
                print(
                    f"  {param:11s} {level:7.2f}  success {row['success_rate']:.2f}  "
                    f"fall {row['fall_rate']:.2f}  progress {row['progress_m']:.2f} m",
                    flush=True,
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures

# Fixed hue per condition (never cycled), colourblind-checked categorical set.
CONDITION_COLORS = {
    "none": "#2a78d6",
    "dr": "#eb6834",
    "rfi": "#1baf7a",
    "rao": "#eda100",
    "erfi_c": "#e87ba4",
    "erfi_50": "#008300",
}
CONDITION_NAMES = {
    "none": "no randomization",
    "dr": "dynamics randomization",
    "rfi": "RFI",
    "rao": "RAO",
    "erfi_c": "ERFI-C",
    "erfi_50": "ERFI-50",
}
_MARKERS = {"none": "o", "dr": "s", "rfi": "^", "rao": "v", "erfi_c": "D", "erfi_50": "P"}


def plot_success_curves(
    results: pd.DataFrame,
    metric: str = "success_rate",
    params: tuple[str, ...] | None = None,
    ncols: int = 3,
) -> Any:
    """Paper Fig. 5 layout: one panel per perturbation, one line per condition.

    Lines are the mean over seeds; the band is min-max over seeds.
    """
    import matplotlib.pyplot as plt

    params = params or tuple(p for p in PROTOCOL if p in set(results["param"]))
    ncols = max(1, min(ncols, len(params)))
    nrows = -(-len(params) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False)
    conditions = [c for c in CONDITION_COLORS if c in set(results["condition"])]

    for ax, param in zip(axes.flat, params):
        sub = results[results["param"] == param]
        for cond in conditions:
            g = sub[sub["condition"] == cond].groupby("level")[metric]
            mean, lo, hi = g.mean(), g.min(), g.max()
            color = CONDITION_COLORS[cond]
            ax.plot(
                mean.index, mean.values, color=color, lw=2, marker=_MARKERS[cond],
                markersize=6, markeredgecolor="white", markeredgewidth=1,
                label=CONDITION_NAMES[cond],
            )
            ax.fill_between(mean.index, lo.values, hi.values, color=color, alpha=0.12, lw=0)
        nominal = PROTOCOL[param][1]
        ax.axvline(nominal, color="#888888", lw=1, ls="--")
        ax.text(nominal, 1.03, "training", color="#666666", fontsize=8, ha="center", va="bottom")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(LABELS[param])
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ax in list(axes.flat)[len(params):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def summary_table(results: pd.DataFrame, metric: str = "success_rate") -> pd.DataFrame:
    """Mean of `metric` over every level and seed, per condition and parameter."""
    return (
        results.groupby(["condition", "param"])[metric].mean().unstack("param")
        .reindex([c for c in CONDITION_COLORS if c in set(results["condition"])])
        .round(3)
    )
