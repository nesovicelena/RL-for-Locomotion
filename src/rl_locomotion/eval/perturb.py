"""Robustness protocol of Campanaro et al. (arXiv:2209.12878, Sec. VII).

The paper deploys each policy with a fixed 0.5 m/s forward command for 8 s and
calls the attempt a success if the robot neither falls nor fails to advance
2.5 m. One simulation parameter is altered at a time. The ANYmal C ranges are
rescaled here to Go1's 12.7 kg (ANYmal C: ~50 kg, 27 kg base):

    payload_kg   extra mass on the trunk           paper: base 22-65 kg (nominal 27)
    push_N       horizontal force on the trunk, 3 s paper: 0-150 N
    friction     floor friction coefficient         paper: 0.2-0.8 (training 0.5;
                                                    ours 0.6 flat / 1.0 rough)
    gravity      m/s^2                              paper: -18 .. -2
    kp_scale     PD position gain multiplier        paper: hardware test at Kp/3

Other robots (the Berkeley Humanoid) state payload and push levels as
fractions of the robot's mass and weight (`EvalSpec.level_fractions`), resolved
against the model at evaluation time; the Go1 numbers above are untouched.

Push direction. `push_N` pushes in a random horizontal direction, as the Go1
study did. `push_N_sagittal` pushes along the initial heading (random sign) and
`push_N_lateral` across it; a biped is far weaker laterally, so the humanoid
study reports the two separately (docs/humanoid_design.md 2.4).

Fall criterion. `fallen` is the task's own termination. With
`EvalSpec.low_base_fraction` set, an episode whose base drops below that
fraction of the spawn height also counts as fallen (`low_base`): the humanoid
terminates only once its torso passes horizontal and its legs do not collide
with the floor, so a kneeling robot would otherwise pass as upright.

ERFI is switched off at evaluation, as in the paper (deployment uses the plain
PD controller), and so are the tasks' built-in disturbances. The terrain is
whatever the run was trained on (`cfg.task`); the trained floor friction is
read from the model rather than assumed, so the friction sweep's "training"
marker is correct on both scenes. Episodes are batched with vmap and stepped
with lax.scan, and the perturbed MJX model is a traced argument, so one compile
per policy covers every parameter level and push axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jp
import numpy as np
import pandas as pd
from mujoco import mjx

from rl_locomotion.envs import erfi

PolicyFn = Callable[[Any, jax.Array], tuple[jax.Array, Any]]

# parameter -> (default levels, value during training). The friction training
# value depends on the scene and is resolved per env by `nominal_value`; the
# 0.6 here is the flat-terrain default kept for backwards compatibility.
PROTOCOL: dict[str, tuple[list[float], float]] = {
    "payload_kg": ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.0),
    "push_N": ([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0], 0.0),
    "friction": ([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 0.6),
    "gravity": ([-2.0, -5.0, -7.0, -9.81, -12.0, -15.0, -18.0], -9.81),
    "kp_scale": ([0.33, 0.5, 0.75, 1.0, 1.25, 1.5], 1.0),
    # Terrain parameters, rough_terrain scenes only (they rewrite the heightfield):
    #   slope_deg          uphill slope of the bowl (see envs/terrain.make_bowl); the
    #                      env's terrain_shape decides whether relief is added on top
    #   terrain_amplitude  peak-to-peak relief of the rocky field, m
    "slope_deg": ([0.0, 10.0, 20.0, 30.0], 0.0),
    "terrain_amplitude": ([0.05, 0.07, 0.08, 0.09, 0.10], 0.05),
}
TERRAIN_PARAMS: tuple[str, ...] = ("slope_deg", "terrain_amplitude")

# Directional variants of the push. Same levels and nominal as push_N.
PUSH_AXES: dict[str, int] = {"push_N": 0, "push_N_sagittal": 1, "push_N_lateral": 2}
ALL_PARAMS: tuple[str, ...] = tuple(PROTOCOL) + ("push_N_sagittal", "push_N_lateral")
# The paper's protocol (the five physical parameters); terrain ones are opt-in.
STANDARD_PARAMS: tuple[str, ...] = ("payload_kg", "push_N", "friction", "gravity", "kp_scale")

# What a fractional level multiplies, per parameter (`EvalSpec.level_fractions`).
_FRACTION_OF = {"payload_kg": "mass", "push_N": "weight", "push_N_sagittal": "weight", "push_N_lateral": "weight"}
G = 9.81


def base_param(param: str) -> str:
    """`push_N_sagittal` -> `push_N`; every other name is its own base."""
    return "push_N" if param in PUSH_AXES else param


def nominal_value(env: Any, param: str) -> float:
    """The value of `param` the policy was trained with, read from the env where it can be."""
    if param == "friction":
        return float(env.mj_model.geom_friction[env._floor_geom_id, 0])
    if param == "gravity":
        return float(env.mj_model.opt.gravity[2])
    if param == "terrain_amplitude":
        return float(getattr(env, "terrain_amplitude", 0.0))
    if param == "slope_deg":
        shape = env._config.get("terrain_shape", "playground") if hasattr(env, "_config") else "playground"
        return float(env._config.get("slope_deg", 0.0)) if shape in ("bowl", "rough_bowl") else 0.0
    return PROTOCOL[base_param(param)][1]


def robot_mass(env: Any) -> float:
    """Total mass in kg (subtree mass of the world body), as the model was built."""
    return float(env.mj_model.body_subtreemass[0])


LABELS = {
    "payload_kg": "payload on trunk (kg)",
    "push_N": "push force on trunk, 3 s (N)",
    "push_N_sagittal": "sagittal push on trunk, 3 s (N)",
    "push_N_lateral": "lateral push on trunk, 3 s (N)",
    "friction": "floor friction coefficient",
    "gravity": "gravity (m/s²)",
    "kp_scale": "Kp multiplier",
    "slope_deg": "uphill slope (deg)",
    "terrain_amplitude": "terrain relief, peak to peak (m)",
}


@dataclass
class EvalSpec:
    command: tuple[float, float, float] = (0.5, 0.0, 0.0)
    duration_s: float = 8.0
    success_distance_m: float = 2.5
    n_episodes: int = 50
    push_start_s: float = 1.0
    push_duration_s: float = 3.0
    params: tuple[str, ...] = STANDARD_PARAMS
    levels: dict[str, list[float]] = field(default_factory=dict)  # override PROTOCOL levels (absolute)
    # Levels as fractions of the robot's mass (payload_kg) or weight m g (push_N*),
    # resolved against the model by `levels_for`. `levels` wins if both are given.
    level_fractions: dict[str, list[float]] = field(default_factory=dict)
    # Base height below this fraction of the spawn height counts as a fall.
    # None keeps the task's termination as the only fall criterion (Go1 studies).
    low_base_fraction: float | None = None
    # Start every episode from the keyframe pose at rest (joints at the default
    # pose, zero velocity), keeping the task's random base position and yaw.
    # Playground's reset scales the joint angles by U(0.5, 1.5) and kicks the
    # base at up to 0.5 m/s, a recovery task in itself for a biped; the paper
    # deploys "always at the same position". Off by default (Go1 studies).
    nominal_reset: bool = False

    def __post_init__(self) -> None:
        unknown = [p for p in self.params if p not in ALL_PARAMS]
        if unknown:
            raise ValueError(f"unknown protocol parameter(s) {unknown}; choose from {list(ALL_PARAMS)}")

    def levels_for(self, param: str, env: Any = None) -> list[float]:
        if param in self.levels:
            return list(self.levels[param])
        if param in self.level_fractions:
            if env is None:
                raise ValueError(f"levels of {param!r} are fractions; an env is needed to resolve them")
            scale = robot_mass(env) * (G if _FRACTION_OF[param] == "weight" else 1.0)
            return [round(float(f) * scale, 4) for f in self.level_fractions[param]]
        return list(PROTOCOL[base_param(param)][0])


def eval_env_config(train_env_cfg: Any, impl: str = "jax") -> Any:
    """Training env config -> evaluation env config (ERFI off, no built-in disturbances)."""
    cfg = erfi.default_config() if train_env_cfg is None else train_env_cfg
    cfg = cfg.copy_and_resolve_references()
    if "task" not in cfg:  # runs from before robot/terrain were part of the config
        cfg.task = "flat_terrain"
    if "robot" not in cfg:
        cfg.robot = "go1"
    if "critic_sees_offset" not in cfg.erfi:  # v1 runs predate the v2 recipe
        cfg.erfi.critic_sees_offset = False
    if "terrain_amplitude" not in cfg:
        cfg.terrain_amplitude = 0.05
    if "terrain_shape" not in cfg:
        cfg.terrain_shape = "playground"
    if "slope_deg" not in cfg:
        cfg.slope_deg = 10.0
    if "hfield_elevation_cap" not in cfg or cfg.hfield_elevation_cap <= 0:
        # fixed elevation so slope / relief can be swept as traced heightfield data;
        # 8 m covers a 30 deg bowl rim (5.2 m) with room to spare
        cfg.hfield_elevation_cap = 8.0
    cfg.erfi.enable = False
    # Go1's `pert_config` velocity kicks, the humanoid's `push_config` pushes.
    cfg[erfi.LAYOUTS[cfg.robot].push_key].enable = False
    cfg.impl = impl
    return cfg


def terrain_model(env: Any, slope_deg: float | None = None, amplitude: float | None = None) -> Any:
    """The env's MJX model with its heightfield rebuilt for another slope and/or relief.

    Follows the env's `terrain_shape`: "playground" -> rocky field at `amplitude`;
    "bowl" -> smooth bowl at `slope_deg`; "rough_bowl" -> bowl plus rocky relief.
    Unspecified values default to the env's own. MJX keeps the heightfield
    elevation static, so the env must have been built with a fixed
    `hfield_elevation_cap` (eval_env_config sets 8 m on rough terrain); the
    terrain then lives entirely in `hfield_data`, a traced model field.
    """
    from rl_locomotion.envs.terrain import build_terrain

    if getattr(env, "task", None) != "rough_terrain" or getattr(env, "_base_hfield", None) is None:
        raise ValueError("terrain parameters need a rough_terrain env (it carries the heightfield)")
    cfg = env._config
    cap = float(cfg.get("hfield_elevation_cap", 0.0))
    if cap <= 0:
        raise ValueError("terrain parameters need hfield_elevation_cap > 0 in the env config (see eval_env_config)")
    shape = cfg.get("terrain_shape", "playground")
    slope = float(cfg.get("slope_deg", 0.0)) if slope_deg is None else float(slope_deg)
    amp = float(cfg.get("terrain_amplitude", 0.05)) if amplitude is None else float(amplitude)
    if slope_deg is not None and shape == "playground":
        shape = "bowl" if amp <= 0 else "rough_bowl"  # a slope asked of a rocky field: put the field on a bowl
    m = env.mj_model
    grid, elevation = build_terrain(
        env._base_hfield, shape, amplitude=amp, slope_deg=slope,
        nrow=int(m.hfield_nrow[0]), ncol=int(m.hfield_ncol[0]), radius_m=float(m.hfield_size[0, 0]),
    )
    if elevation > cap + 1e-9:
        raise ValueError(f"terrain rises {elevation:.2f} m, above hfield_elevation_cap={cap}")
    model = env.mjx_model
    start = int(m.hfield_adr[0]); n = grid.size
    data = jp.asarray(model.hfield_data).at[start : start + n].set(jp.asarray(grid.ravel() * (elevation / cap), dtype=jp.float32))
    return model.tree_replace({"hfield_data": data})


def perturbed_model(env: Any, param: str, level: float) -> Any:
    """A copy of the env's MJX model with one parameter altered."""
    if param == "slope_deg":
        return terrain_model(env, slope_deg=level)
    if param == "terrain_amplitude":
        return terrain_model(env, amplitude=level)
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
    if param in PUSH_AXES:
        return model  # applied as an external force during the rollout
    raise ValueError(f"unknown perturbation {param!r}")


def nominal_start(env: Any, state: Any) -> Any:
    """Replace the reset's perturbed joints and velocities by the keyframe pose at rest.

    Base position and yaw from the reset are kept. The ERFI history is refilled
    from the new reading instead of rolling the perturbed one in; the base
    observation redraws its sensor noise. Uses `env.mjx_model`, so inside a
    protocol rollout it sees the perturbed model.
    """
    default_pose = jp.array(env.mj_model.keyframe("home").qpos[7:])
    data = state.data.replace(
        qpos=state.data.qpos.at[7:].set(default_pose),
        qvel=jp.zeros_like(state.data.qvel),
        ctrl=default_pose,
    )
    data = mjx.forward(env.mjx_model, data)
    info = dict(state.info)
    info.pop("joint_pos_hist", None)
    info.pop("joint_vel_hist", None)
    obs = env._get_obs(data, info, *env._obs_extra_args(data))
    return state.replace(data=data, obs=obs, info=info)


def make_batched_rollout(env: Any, policy: PolicyFn, spec: EvalSpec):
    """Compile `run(model, keys, push_force, push_axis) -> per-episode results` for this env+policy.

    `push_axis` is the code from PUSH_AXES (0 random direction, 1 sagittal,
    2 lateral); it is traced, so one compile serves all three.
    """
    n_steps = int(round(spec.duration_s / env.dt))
    push_start = int(round(spec.push_start_s / env.dt))
    push_end = push_start + int(round(spec.push_duration_s / env.dt))
    command = jp.array(spec.command)
    torso = env._torso_body_id
    nbody = env.mjx_model.nbody
    low_z = -jp.inf if spec.low_base_fraction is None else spec.low_base_fraction * env.spawn_height

    def episode(key: jax.Array, push_force: jax.Array, push_axis: jax.Array) -> dict[str, jax.Array]:
        key, k_reset, k_dir = jax.random.split(key, 3)
        state = env.reset(k_reset)
        if spec.nominal_reset:
            state = nominal_start(env, state)
        state.info["command"] = command

        # Heading = body x-axis projected on the ground; progress is measured along it.
        x_axis = state.data.xmat[torso][:, 0]
        heading = x_axis[:2] / (jp.linalg.norm(x_axis[:2]) + 1e-6)
        pos0 = state.data.xpos[torso][:2]

        # One draw serves all three axes: the angle for `random`, its sign for the others.
        angle = jax.random.uniform(k_dir, minval=0.0, maxval=2 * jp.pi)
        sign = jp.where(angle < jp.pi, 1.0, -1.0)
        lateral = jp.array([-heading[1], heading[0]])
        dir_xy = jp.select(
            [push_axis == 1, push_axis == 2],
            [sign * heading, sign * lateral],
            jp.array([jp.cos(angle), jp.sin(angle)]),
        )
        push_dir = jp.array([dir_xy[0], dir_xy[1], 0.0])

        def body(carry, t):
            state, key, fallen, low, progress, err_sum, alive_steps = carry
            key, k_act = jax.random.split(key)
            action, _ = policy(state.obs, k_act)

            active = (t >= push_start) & (t < push_end)
            xfrc = jp.zeros((nbody, 6)).at[torso, :3].set(push_force * push_dir * active)
            state = state.replace(data=state.data.replace(xfrc_applied=xfrc))

            state = env.step(state, action)
            state.info["command"] = command

            now_low = low | (state.data.xpos[torso][2] < low_z)
            now_fallen = fallen | (state.done > 0) | now_low
            cur = jp.dot(state.data.xpos[torso][:2] - pos0, heading)
            progress = jp.where(now_fallen, progress, cur)  # freeze at the fall
            v = env.get_local_linvel(state.data)
            err = jp.sum(jp.square(command[:2] - v[:2]))
            err_sum = jp.where(now_fallen, err_sum, err_sum + err)
            alive_steps = jp.where(now_fallen, alive_steps, alive_steps + 1)
            return (state, key, now_fallen, now_low, progress, err_sum, alive_steps), None

        init = (state, key, jp.zeros((), bool), jp.zeros((), bool), jp.zeros(()), jp.zeros(()), jp.zeros((), jp.int32))
        (_, _, fallen, low, progress, err_sum, alive_steps), _ = jax.lax.scan(body, init, jp.arange(n_steps))
        success = (~fallen) & (progress >= spec.success_distance_m)
        return {
            "success": success,
            "fallen": fallen,
            "low_base": low,
            "progress_m": progress,
            "tracking_rmse": jp.sqrt(err_sum / n_steps),
            "tracking_rmse_alive": jp.sqrt(err_sum / jp.maximum(alive_steps, 1)),
        }

    def run(model: Any, keys: jax.Array, push_force: jax.Array, push_axis: jax.Array) -> dict[str, jax.Array]:
        # Swap the traced model in for the duration of tracing only. Cached
        # executions never re-enter this Python body.
        saved = env._mjx_model
        env._mjx_model = model
        try:
            return jax.vmap(episode, in_axes=(0, None, None))(keys, push_force, push_axis)
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
    task = getattr(env, "task", "flat_terrain")
    robot = getattr(env, "robot", "go1")
    mass = robot_mass(env)
    for param in spec.params:
        nominal = nominal_value(env, param)
        axis = PUSH_AXES.get(param, 0)
        for level in spec.levels_for(param, env):
            model = perturbed_model(env, param, level)
            push = jp.asarray(level if param in PUSH_AXES else 0.0, dtype=jp.float32)
            out = jax.device_get(run(model, keys, push, jp.asarray(axis, dtype=jp.int32)))
            row = {
                **(meta or {}),
                "robot": robot,
                "task": task,
                "robot_mass_kg": mass,
                "param": param,
                "level": level,
                "nominal": nominal,
                "success_rate": float(np.mean(out["success"])),
                "fall_rate": float(np.mean(out["fallen"])),
                "low_base_rate": float(np.mean(out["low_base"])),
                "progress_m": float(np.mean(out["progress_m"])),
                "tracking_rmse": float(np.mean(out["tracking_rmse"])),
                "tracking_rmse_alive": float(np.mean(out["tracking_rmse_alive"])),
                "n_episodes": spec.n_episodes,
            }
            rows.append(row)
            if verbose:
                print(
                    f"  {param:16s} {level:7.2f}  success {row['success_rate']:.2f}  "
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

    params = params or tuple(p for p in ALL_PARAMS if p in set(results["param"]))
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
        nominal = float(sub["nominal"].iloc[0]) if "nominal" in sub else PROTOCOL[base_param(param)][1]
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
