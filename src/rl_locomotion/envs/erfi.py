r"""Extended Random Force Injection (ERFI) on Playground-style joystick tasks.

Campanaro et al., "Learning and Deploying Robust Locomotion Policies with
Minimal Dynamics Randomization" (arXiv:2209.12878).

Actions stay reference joint positions — the perturbation is a torque added
on top of the actuator output, at the joint DOFs:

    tau_j = Kp(q*_j - q_j) - Kd qdot_j  +  tau_r,j  +  tau_o,j
            \_________ position actuator _________/    \___ qfrc_applied ___/

    RFI  tau_r ~ U(-r_lim, r_lim)   resampled every control step (50 Hz), or
                                    every physics substep with `erfi.per_substep`
    RAO  tau_o ~ U(-o_lim, o_lim)   sampled once per episode

RFI rate. The paper draws tau_r at the *impedance control* frequency, which is
higher than the policy's. Here the mixin sits above Playground's `step`, whose
substep loop is a closed lax.scan, so the only free injection point is
`qfrc_applied` before the call -- which holds for the whole control interval and
ties the RFI rate to `ctrl_dt` (50 Hz). Measured on Go1, a 0.17 rad step at the
knee has a 32 ms time constant, so a 20 ms draw is 0.62 of it: the joint
partially tracks each draw instead of averaging it, which makes RFI behave
partly like a short RAO. `erfi.per_substep` replaces that loop (see
`_substep_rfi`) and redraws tau_r every physics substep -- 250 Hz on the
quadrupeds, 500 Hz on the humanoid. Off by default, so every run in
experiments/redo stays reproducible.

Modes:
    "rfi"      tau_r only
    "rao"      tau_o only
    "erfi_c"   both, every episode
    "erfi_50"  each episode is RFI or RAO with probability 1/2 (paper default)

Observation (paper Sec. VI-B, the blind A1 setup), with H = history_len and
nu joints:

    gravity (3) | base linvel (3) | gyro (3) | joint-pos-error history (nu H)
    | joint-vel history (nu H) | previous action (nu) | command (3) | [phase]

H = 7 gives the paper's 192 dimensions on Go1/A1; H = 1 is Playground's 48-dim
state reordered. The humanoid tasks end with a 4-entry gait clock (`phase`)
their reward depends on; it is kept as a tail (196 dimensions at H = 7 on the
Berkeley Humanoid). The critic keeps Playground's privileged state so an
asymmetric critic remains available.

The joint-position history holds q - q_default by default (Playground's
entry; what v1 and v2 were trained with). With `cfg.history_target_error`
it holds the paper's tracking error q* - q instead.

Robots. `cfg.robot` selects the model and the task layout (`ROBOTS`, `LAYOUTS`):
"go1" (Playground's Go1JoystickFlatTerrain / RoughTerrain), "a1" (our port of
the same task to Menagerie's Unitree A1, the robot of the paper's blind
experiment; see envs/a1/), "bh" (Playground's Berkeley Humanoid joystick task;
see envs/bh/ and docs/humanoid_design.md). The ERFI logic below only needs the
joint DOFs to follow the six free-base DOFs and the state to hold the joint
readings in one contiguous block at a known offset.

Torque limits. `cfg.erfi.rfi_lim` / `rao_lim` are either one number (Go1/A1
studies: 2.5 Nm on every joint) or a per-joint list of `nu` numbers (the
humanoid, where stance torques differ by an order of magnitude between joints).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import registry as pg_registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion import register_environment
from mujoco_playground._src.locomotion.go1 import joystick

from rl_locomotion.envs import bh as bh_pkg
from rl_locomotion.envs.a1 import A1Joystick
from rl_locomotion.envs.bh import BerkeleyHumanoidJoystick

ENV_NAME = "Go1JoystickERFI"
MODES = ("rfi", "rao", "erfi_c", "erfi_50")

# Playground scenes for the joystick task. The terrain is part of the config
# (`cfg.task`) rather than a constructor argument, so that a run's
# env_config.json fully determines the environment and evaluation can never
# silently rebuild a rough-terrain policy on flat ground.
#   flat_terrain   plane, floor friction 0.6, keyframe height 0.278 m (A1: 0.27, BH: 0.515)
#   rough_terrain  20 x 20 m heightfield, 5 cm peak-to-peak (std 1.45 cm),
#                  floor friction 1.0, keyframe height 0.35 m (A1: 0.34, BH: 0.56)
TASKS = ("flat_terrain", "rough_terrain")


@dataclass(frozen=True)
class RobotLayout:
    """What the mixin needs to know about a robot's Playground task.

    joint_pos_start   offset of `joint_angles - default_pose` in Playground's state;
                      joint velocities follow immediately (nu entries each)
    phase_dim         entries of the gait-clock tail at the end of the state (0 if none)
    push_key          config node of the task's built-in disturbance
                      (Go1 `pert_config` velocity kicks, humanoid `push_config`)
    playground_env    Playground env name per task, for the domain randomizer and
                      the tuned PPO config
    base_config       the task's own default_config()
    """

    joint_pos_start: int
    phase_dim: int
    push_key: str
    playground_env: dict[str, str]
    base_config: Callable[[], config_dict.ConfigDict]


# Robot models. Same rule as the terrain: part of the config, recorded per run.
#   go1  Unitree Go1, 12.74 kg, limits 23.7 Nm hip/thigh, 35.55 Nm knee
#   a1   Unitree A1, 12.45 kg, limit 33.5 Nm on all joints (the paper's blind robot)
#   bh   Berkeley Humanoid, 16.06 kg, 12 joints, joint clamps 5-30 Nm (envs/bh/)
_GO1_LAYOUT = RobotLayout(
    # Go1 state: linvel 0:3 | gyro 3:6 | gravity 6:9 | joints 9:21 | joint_vel 21:33
    # | last_act 33:45 | command 45:48
    joint_pos_start=9,
    phase_dim=0,
    push_key="pert_config",
    playground_env={"flat_terrain": "Go1JoystickFlatTerrain", "rough_terrain": "Go1JoystickRoughTerrain"},
    base_config=joystick.default_config,
)
LAYOUTS: dict[str, RobotLayout] = {
    "go1": _GO1_LAYOUT,
    # Same task code on a different model, so the same layout and randomizer.
    "a1": _GO1_LAYOUT,
    # Berkeley state: linvel 0:3 | gyro 3:6 | gravity 6:9 | command 9:12 | joints 12:24
    # | joint_vel 24:36 | last_act 36:48 | phase 48:52
    "bh": RobotLayout(
        joint_pos_start=12,
        phase_dim=4,
        push_key="push_config",
        playground_env={
            "flat_terrain": "BerkeleyHumanoidJoystickFlatTerrain",
            "rough_terrain": "BerkeleyHumanoidJoystickRoughTerrain",
        },
        base_config=bh_pkg.default_config,
    ),
}
ROBOTS = tuple(LAYOUTS)

# The six training conditions of the study. `randomize` toggles Playground's
# dynamics randomizer (friction, masses, CoM, armature); `mode` the ERFI scheme.
CONDITIONS: dict[str, dict[str, Any]] = {
    "none": dict(erfi=False, mode="erfi_50", randomize=False),
    "dr": dict(erfi=False, mode="erfi_50", randomize=True),
    "rfi": dict(erfi=True, mode="rfi", randomize=False),
    "rao": dict(erfi=True, mode="rao", randomize=False),
    "erfi_c": dict(erfi=True, mode="erfi_c", randomize=False),
    "erfi_50": dict(erfi=True, mode="erfi_50", randomize=False),
}


def playground_env_name(robot: str, task: str = "flat_terrain") -> str:
    """Playground's registered name of the task the robot's ERFI env is built on."""
    return LAYOUTS[robot].playground_env[task]


def default_config(robot: str = "go1") -> config_dict.ConfigDict:
    if robot not in LAYOUTS:
        raise ValueError(f"robot must be one of {ROBOTS}, got {robot!r}")
    layout = LAYOUTS[robot]
    cfg = layout.base_config()
    cfg.robot = robot
    cfg.task = "flat_terrain"
    # Peak-to-peak relief of the rough-terrain heightfield in metres. Playground's
    # scene is 0.05; the terrain curriculum trains stages at 0, 0.015, 0.03, 0.05.
    # Ignored on flat_terrain. Applied by rescaling hfield_size[2] after the
    # model is built, so the same heightfield shape is used at every amplitude.
    cfg.terrain_amplitude = 0.05
    # Shape of the rough-terrain heightfield: "playground" (the rocky field
    # shipped with Go1JoystickRoughTerrain, relief = terrain_amplitude) or
    # "bowl" (flat 1 m disc, then a constant uphill slope of `slope_deg` in
    # every direction; see envs/terrain.make_bowl), or "rough_bowl" (the bowl
    # with the rocky relief of `terrain_amplitude` added on top). Ignored on
    # flat_terrain.
    cfg.terrain_shape = "playground"
    cfg.slope_deg = 10.0
    # If > 0, the heightfield's elevation is fixed at this many metres and the
    # terrain is encoded in the normalised height data instead. MJX keeps the
    # elevation static but the data traced, so this is what lets the evaluation
    # protocol sweep slope and relief inside one compiled rollout. 0 = off
    # (elevation = the terrain's own height, as Playground does).
    cfg.hfield_elevation_cap = 0.0
    # Paper: 7-step history of joint position errors and joint velocities.
    # (Berkeley's config already carries an unused `history_len = 1`; overwritten.)
    cfg.history_len = 7
    # What the joint-position history holds.
    #   False  q - q_default, Playground's state entry (v1 and v2 runs).
    #   True   q* - q, the paper's tracking error, with q* the target the PD
    #          controller is holding when the observation is taken
    #          (q_default + action * action_scale of the action applied in
    #          this step; at reset the target is the keyframe pose).
    # Changes the policy input, so it is a new recipe, not a drop-in change.
    cfg.history_target_error = False
    if robot == "bh":
        # Playground trains the humanoid with random velocity pushes on. That is
        # a disturbance-training method of its own and would blur the `none`
        # condition, so the study switches it off in every condition
        # (docs/humanoid_design.md 4.6). Evaluation disables it regardless.
        cfg.push_config.enable = False
        # Extra termination: base height below this (m) ends the episode. The
        # task itself terminates only once the torso passes horizontal, and the
        # scene collides feet only, so a buckled robot would otherwise sink
        # through the floor and keep training. Half the flat spawn height
        # (0.515 m); the rough scene spawns at 0.56 m over up to 0.05 m of
        # relief, so the same value leaves 0.2 m of margin there. 0 disables.
        cfg.min_base_height = 0.5 * 0.515
        # Playground's `feet_slip` cost multiplies the *base* velocity by the
        # contact flags, i.e. it penalises walking speed during stance. The ERFI
        # class replaces it with the feet's own velocity from the foot sensors
        # (docs/humanoid_design.md 8.1). Set the scale to 0 to drop the term.
        rfi_lim: Any = list(bh_pkg.PROVISIONAL_TORQUE_LIMIT)
        rao_lim: Any = list(bh_pkg.PROVISIONAL_TORQUE_LIMIT)
    else:
        # Nm. The paper uses 20 Nm on the 50 kg ANYmal C, about half a joint's
        # stance torque. Go1 and A1 are both ~12.5 kg, so the study uses 2.5 Nm
        # for both (set by the experiment config; 7.0 here is the historical
        # default that turned out too large, see the ERFI report).
        rfi_lim, rao_lim = 7.0, 7.0
    cfg.erfi = config_dict.create(
        enable=True,
        mode="erfi_50",
        # v2 recipe: append the episode's RAO offset (nu) and the RFI flag (1)
        # to the critic's privileged state (123 -> 136 on Go1). The policy input
        # is untouched. Lets the asymmetric critic explain return variance
        # caused by the hidden per-episode offset, which otherwise makes the
        # advantage of "start walking" noisy and stalls RAO/ERFI runs in the
        # standing optimum. v1 studies were trained with False.
        critic_sees_offset=False,
        # Redraw tau_r every physics substep instead of every control step, by
        # replacing Playground's substep loop (`_substep_rfi`). True is the
        # paper's rate (250 Hz on the quadrupeds, 500 Hz on the humanoid);
        # False is what every run in experiments/redo was trained with.
        per_substep=False,
        # One number for every joint, or a list with one entry per joint
        # (`set_torque_limits`). The humanoid's provisional vector is 10 % of
        # each joint's actuator-force clamp; the study replaces it with a
        # measured one (scripts/measure_stance_torque.py).
        rfi_lim=rfi_lim,
        rao_lim=rao_lim,
    )
    return cfg


def set_torque_limits(cfg: config_dict.ConfigDict, rfi_lim: Any, rao_lim: Any) -> None:
    """Write scalar or per-joint limits into `cfg.erfi`, whatever type is there now.

    ConfigDict locks the type of a field on first assignment; the same field is
    a float for Go1/A1 and a list for the humanoid, so bypass the check here.
    """
    with cfg.erfi.ignore_type():
        cfg.erfi.rfi_lim = _as_limit(rfi_lim)
        cfg.erfi.rao_lim = _as_limit(rao_lim)


def _as_limit(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return float(value)


def condition_config(name: str, **overrides: Any) -> config_dict.ConfigDict:
    """Default config for one of the study's training conditions."""
    if name not in CONDITIONS:
        raise ValueError(f"unknown condition {name!r}; choose from {list(CONDITIONS)}")
    robot = overrides.pop("robot", "go1")
    if robot not in LAYOUTS:
        raise ValueError(f"robot must be one of {ROBOTS}, got {robot!r}")
    cfg = default_config(robot)
    cfg.erfi.enable = CONDITIONS[name]["erfi"]
    cfg.erfi.mode = CONDITIONS[name]["mode"]
    for k, v in overrides.items():
        cfg[k] = v
    if cfg.task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {cfg.task!r}")
    if cfg.task == "rough_terrain":
        # More contacts against the heightfield than against a plane. Playground
        # uses naconmax 8*8192 and njmax 60; Warp 1.16 overflowed njmax=60 in
        # training ("nefc overflow - please increase njmax to 64"), and an
        # overflow silently drops constraint rows, so use a wide margin. The
        # extra rows cost a little memory, not correctness.
        cfg.naconmax = 8 * 8192
        cfg.njmax = 128
    return cfg


def uses_domain_randomization(name: str) -> bool:
    return CONDITIONS[name]["randomize"]


class _ERFIMixin:
    """ERFI torque perturbation + history observation, on top of a Playground joystick task.

    Must come first in the MRO; the base class is the robot-specific Joystick.
    """

    ROBOT: str = ""

    def __init__(self, task=None, config=None, config_overrides=None):
        cfg = default_config(self.ROBOT) if config is None else config
        # The terrain and robot live in the config. A `task` argument is accepted
        # for API compatibility with Playground but must agree with `cfg.task`.
        cfg_task = cfg.get("task", "flat_terrain")
        cfg_robot = cfg.get("robot", "go1")
        if task is not None and task != cfg_task:
            raise ValueError(f"task={task!r} disagrees with config.task={cfg_task!r}")
        if cfg_task not in TASKS:
            raise ValueError(f"config.task must be one of {TASKS}, got {cfg_task!r}")
        if cfg_robot != self.ROBOT:
            raise ValueError(f"{type(self).__name__} is the {self.ROBOT!r} env; config.robot is {cfg_robot!r}")
        self._layout = LAYOUTS[self.ROBOT]
        # Playground's Joystick.__init__ overwrites naconmax/njmax for rough
        # terrain with its own (too small) values. Remember ours and restore
        # them afterwards; make_data reads them at reset time, so this is
        # enough. Never lower what Playground asked for.
        wanted_naconmax, wanted_njmax = cfg.naconmax, cfg.njmax
        super().__init__(task=cfg_task, config=cfg, config_overrides=config_overrides)
        self._config.naconmax = max(self._config.naconmax, wanted_naconmax)
        self._config.njmax = max(self._config.njmax, wanted_njmax)
        self._task = cfg_task
        # The scene's own heightfield (Playground's rocky relief) as a normalised
        # grid + elevation, kept so the evaluation protocol can rebuild any
        # terrain shape / slope / relief from it (eval/perturb.terrain_model).
        self._base_hfield = None
        if cfg_task == "rough_terrain":
            from rl_locomotion.envs.terrain import height_map as _hm

            base = _hm(self._mj_model)
            self._base_hfield = (
                (base.heights / base.elevation if base.elevation > 0 else base.heights).astype("float32"),
                float(base.elevation),
            )
            shape = self._config.get("terrain_shape", "playground")
            from rl_locomotion.envs.terrain import apply_heightfield, build_terrain

            amp = float(self._config.get("terrain_amplitude", 0.05))
            if amp < 0:
                raise ValueError("terrain_amplitude must be >= 0")
            if shape not in ("playground", "bowl", "rough_bowl"):
                raise ValueError(f"terrain_shape must be 'playground', 'bowl' or 'rough_bowl', got {shape!r}")
            m = self._mj_model
            grid, elevation = build_terrain(
                self._base_hfield, shape, amplitude=amp, slope_deg=float(self._config.get("slope_deg", 10.0)),
                nrow=int(m.hfield_nrow[0]), ncol=int(m.hfield_ncol[0]), radius_m=float(m.hfield_size[0, 0]),
            )
            cap = float(self._config.get("hfield_elevation_cap", 0.0))
            if cap > 0:
                if elevation > cap + 1e-9:
                    raise ValueError(f"terrain rises {elevation:.2f} m, above hfield_elevation_cap={cap}")
                grid, elevation = (grid * (elevation / cap)).astype("float32"), cap
            if shape != "playground" or cap > 0 or abs(elevation - float(m.hfield_size[0, 2])) > 1e-9:
                apply_heightfield(m, grid, elevation)
                self._mjx_model = mjx.put_model(m, impl=self._config.impl)
        mode = self._config.erfi.mode
        if mode not in MODES:
            raise ValueError(f"erfi.mode must be one of {MODES}, got {mode!r}")
        if self._config.history_len < 1:
            raise ValueError("history_len must be >= 1")
        # Joint DOFs sit after the 6 free-base DOFs.
        nu, nv = self.mjx_model.nu, self.mjx_model.nv
        if nv - nu != 6:
            raise ValueError(f"expected six free-base DOFs before the joints, got nv - nu = {nv - nu}")
        self._joint_dof_start = nv - nu
        s = self._layout.joint_pos_start
        self._joint_pos_slice = slice(s, s + nu)
        self._joint_vel_slice = slice(s + nu, s + 2 * nu)
        self._rfi_lim = self._limit_vector(self._config.erfi.rfi_lim, "rfi_lim")
        self._rao_lim = self._limit_vector(self._config.erfi.rao_lim, "rao_lim")

    def _limit_vector(self, value: Any, name: str) -> jax.Array:
        """Scalar or per-joint limit -> array broadcastable against (nu,)."""
        arr = jp.asarray(value, dtype=jp.float32)
        if arr.ndim == 0:
            return arr
        if arr.shape != (self.mjx_model.nu,):
            raise ValueError(f"erfi.{name} must be a number or {self.mjx_model.nu} numbers, got shape {arr.shape}")
        return arr

    @property
    def task(self) -> str:
        return self._task

    @property
    def robot(self) -> str:
        return self.ROBOT

    @property
    def layout(self) -> RobotLayout:
        return self._layout

    @property
    def rfi_lim(self) -> jax.Array:
        return self._rfi_lim

    @property
    def rao_lim(self) -> jax.Array:
        return self._rao_lim

    @property
    def terrain_amplitude(self) -> float:
        """Peak-to-peak relief of the rocky field in metres (0.0 on flat terrain or a smooth bowl).

        On bowl shapes the heightfield's elevation is the rim height, so the
        relief is read from the config instead.
        """
        if self._task != "rough_terrain":
            return 0.0
        shape = self._config.get("terrain_shape", "playground")
        if shape == "bowl":
            return 0.0
        return float(self._config.get("terrain_amplitude", 0.05))

    @property
    def floor_friction(self) -> float:
        """Sliding friction of the floor geom as trained (0.6 flat, 1.0 rough)."""
        return float(self.mj_model.geom_friction[self._floor_geom_id, 0])

    @property
    def total_mass(self) -> float:
        """Mass of the whole robot in kg (subtree mass of the world body)."""
        return float(self.mj_model.body_subtreemass[0])

    @property
    def spawn_height(self) -> float:
        """Base height of the task's `home` keyframe in metres."""
        return float(self.mj_model.keyframe("home").qpos[2])

    # ------------------------------------------------------------------ ERFI

    def _sample_erfi(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Per-episode: the RAO offset, and which channels this episode uses."""
        cfg = self._config.erfi
        # rng, k_off and k_mode are split here so the caller can use the leftover rng for other draws.
        # k_mode is used to decide whether to use RFI or RAO in the "erfi_50" mode.
        rng, k_off, k_mode = jax.random.split(rng, 3)

        # minval/maxval broadcast, so a per-joint limit vector works unchanged.
        tau_o = jax.random.uniform(
            k_off, (self.mjx_model.nu,), minval=-self._rao_lim, maxval=self._rao_lim
        )

        if cfg.mode == "rfi":
            use_rfi, use_rao = 1.0, 0.0
        elif cfg.mode == "rao":
            use_rfi, use_rao = 0.0, 1.0
        elif cfg.mode == "erfi_c":
            use_rfi, use_rao = 1.0, 1.0
        else:  # erfi_50 — this episode is one or the other
            is_rfi = jax.random.bernoulli(k_mode)
            use_rfi = is_rfi.astype(jp.float32)
            use_rao = 1.0 - use_rfi

        return tau_o * use_rao, jp.asarray(use_rfi, dtype=jp.float32)

    @contextlib.contextmanager
    def _substep_rfi(self, key: jax.Array, offset: jax.Array, use_rfi: jax.Array):
        """Swap Playground's substep loop for one that redraws tau_r every substep.

        `Joystick.step` calls `mjx_env.step(model, data, ctrl, n_substeps)`, a
        closed lax.scan that only refreshes `ctrl`; `qfrc_applied` is carried in
        `data` and therefore held for the whole control interval. Replacing the
        module attribute for the duration of the call is the same trick
        `eval/perturb.make_batched_rollout` uses for `env._mjx_model`: the parent
        looks the function up on the module at call time, and both the quadruped
        and the humanoid task go through this one function.

        The replacement keeps `ctrl` handling identical and adds one draw per
        substep, so `per_substep=False` and the original are bit-identical.
        """
        original = mjx_env.step
        nu, nv, start = self.mjx_model.nu, self.mjx_model.nv, self._joint_dof_start
        lim = self._rfi_lim

        def step_with_substep_rfi(model, data, action, n_substeps=1):
            def single_step(carry, _):
                data, k = carry
                k, k_tau = jax.random.split(k)
                tau_r = jax.random.uniform(k_tau, (nu,), minval=-lim, maxval=lim)
                qfrc = jp.zeros(nv).at[start:].set(offset + tau_r * use_rfi)
                data = data.replace(ctrl=action, qfrc_applied=qfrc)
                return (mjx.step(model, data), k), None

            (data, _), _ = jax.lax.scan(single_step, (data, key), (), n_substeps)
            return data

        mjx_env.step = step_with_substep_rfi
        try:
            yield
        finally:
            mjx_env.step = original

    # ----------------------------------------------------------- observation

    def _obs_extra_args(self, data: Any) -> tuple:
        """Extra positional arguments the base task's `_get_obs` takes (none for Go1)."""
        del data
        return ()

    def _get_obs(self, data: Any, info: dict[str, Any], *args: Any) -> dict[str, jax.Array]:
        obs = super()._get_obs(data, info, *args)
        pg = obs["state"]
        joint_pos, joint_vel = pg[self._joint_pos_slice], pg[self._joint_vel_slice]
        if self._config.get("history_target_error", False):
            # q* - q = (q_default + a * scale) - q = a * scale - (q - q_default).
            # `applied_act` is the action of the step being observed; zeros at
            # reset, where ctrl is the keyframe pose.
            nu = self.mjx_model.nu
            applied = info.get("applied_act", jp.zeros(nu))
            joint_pos = applied * self._config.action_scale - joint_pos

        H = self._config.history_len
        if "joint_pos_hist" not in info:  # first call, from reset()
            # fill every slot with current value
            info["joint_pos_hist"] = jp.tile(joint_pos, (H, 1))
            info["joint_vel_hist"] = jp.tile(joint_vel, (H, 1))
        else:  # newest reading first
            # every slot moves one slot older
            info["joint_pos_hist"] = jp.roll(info["joint_pos_hist"], 1, axis=0).at[0].set(joint_pos)
            info["joint_vel_hist"] = jp.roll(info["joint_vel_hist"], 1, axis=0).at[0].set(joint_vel)

        parts = [
            pg[6:9],  # gravity in body frame (orientation, yaw-free)
            pg[0:3],  # base linear velocity
            pg[3:6],  # gyro
            info["joint_pos_hist"].ravel(),
            info["joint_vel_hist"].ravel(),
            info["last_act"],
            info["command"],
        ]
        if self._layout.phase_dim:
            # The task's gait clock (cos/sin per foot); its reward depends on it.
            parts.append(pg[pg.shape[0] - self._layout.phase_dim :])
        state = jp.hstack(parts)
        privileged = obs["privileged_state"]
        if self._config.erfi.get("critic_sees_offset", False):
            # Zeros on the very first call from reset(); reset() recomputes the
            # observation once the episode's offset has been drawn.
            nu = self.mjx_model.nu
            offset = info.get("erfi_offset", jp.zeros(nu))
            use_rfi = info.get("erfi_use_rfi", jp.zeros(()))
            privileged = jp.hstack([privileged, offset, jp.reshape(use_rfi, (1,))])
        return {"state": state, "privileged_state": privileged}

    # ----------------------------------------------------------- reset/step

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key = jax.random.split(rng)
        state = super().reset(rng)
        # Present from reset on, so the state pytree has the same structure
        # after reset and after step (lax.scan and the auto-reset need that).
        state.info["applied_act"] = jp.zeros(self.mjx_model.nu)
        tau_o, use_rfi = self._sample_erfi(key)
        if not self._config.erfi.enable:
            tau_o, use_rfi = jp.zeros_like(tau_o), jp.zeros(())
        state.info["erfi_offset"] = tau_o
        state.info["erfi_use_rfi"] = use_rfi
        if self._config.erfi.get("critic_sees_offset", False):
            # The history is already filled with the current reading, so the
            # roll inside _get_obs leaves it unchanged; only the critic's extra
            # entries change.
            obs = self._get_obs(state.data, state.info, *self._obs_extra_args(state.data))
            state = state.replace(obs=obs)
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg = self._config.erfi
        per_substep = cfg.enable and cfg.get("per_substep", False)
        key = None
        if cfg.enable:
            rng, key = jax.random.split(state.info["rng"])
            state.info["rng"] = rng
            # With per_substep the draw happens inside the substep loop, so only
            # the episode offset goes in here; it is what the first substep would
            # otherwise start from, and RAO-only episodes never enter the loop's
            # RFI term anyway (use_rfi is 0).
            tau = state.info["erfi_offset"]
            if not per_substep:
                tau_r = jax.random.uniform(
                    key, (self.mjx_model.nu,), minval=-self._rfi_lim, maxval=self._rfi_lim
                )
                tau = tau + tau_r * state.info["erfi_use_rfi"]
            qfrc = jp.zeros(self.mjx_model.nv).at[self._joint_dof_start :].set(tau)
            state = state.replace(data=state.data.replace(qfrc_applied=qfrc))

        # The parent computes the observation before it records `last_act`, so
        # expose the action whose target the PD controller holds right now.
        state.info["applied_act"] = action

        if per_substep:
            # Redraw tau_r at the physics rate; see `_substep_rfi`.
            with self._substep_rfi(key, state.info["erfi_offset"], state.info["erfi_use_rfi"]):
                state = super().step(state, action)
        else:
            # qfrc_applied persists across the substeps inside mjx_env.step, so the
            # parent's step applies tau for the whole control interval.
            state = super().step(state, action)
        done = state.done.astype(bool)

        if cfg.enable:
            # Brax-style auto-reset (the default in wrap_for_brax_training)
            # restores data/obs on `done` but keeps `info`. If the per-episode
            # draw only happened in reset(), every env slot would keep its first
            # RAO offset for the whole run. Redraw whenever the episode ends.
            rng, key = jax.random.split(state.info["rng"])
            state.info["rng"] = rng
            new_offset, new_use_rfi = self._sample_erfi(key)
            state.info["erfi_offset"] = jp.where(done, new_offset, state.info["erfi_offset"])
            state.info["erfi_use_rfi"] = jp.where(done, new_use_rfi, state.info["erfi_use_rfi"])

            # Clear, so a stale perturbation never leaks into reward/termination
            # readouts or the next step's forward pass.
            zero = jp.zeros(self.mjx_model.nv)
            state = state.replace(data=state.data.replace(qfrc_applied=zero))

        # Same auto-reset caveat for the history: flatten it on episode end so
        # the next episode does not start with the previous one's tail.
        for k in ("joint_pos_hist", "joint_vel_hist"):
            hist = state.info[k]
            state.info[k] = jp.where(done, jp.tile(hist[0], (hist.shape[0], 1)), hist)
        return state


class Go1JoystickERFI(_ERFIMixin, joystick.Joystick):
    """Unitree Go1 joystick with ERFI torque perturbations and a history observation."""

    ROBOT = "go1"


class A1JoystickERFI(_ERFIMixin, A1Joystick):
    """Unitree A1 joystick with ERFI torque perturbations and a history observation."""

    ROBOT = "a1"


class BerkeleyHumanoidJoystickERFI(_ERFIMixin, BerkeleyHumanoidJoystick):
    """Berkeley Humanoid joystick with ERFI torque perturbations and a history observation."""

    ROBOT = "bh"

    def _get_termination(self, data: Any) -> jax.Array:
        # Task: gravity z < 0 or NaN. Plus: base below `min_base_height` (see
        # default_config). A true terminal for Brax (it arrives through
        # `state.done`, not the episode wrapper's truncation), so no value is
        # bootstrapped from a collapsed pose.
        done = super()._get_termination(data)
        h = float(self._config.get("min_base_height", 0.0))
        if h > 0:
            done = done | (data.qpos[2] < h)
        return done

    def _cost_feet_slip(self, data: Any, contact: jax.Array, info: dict[str, Any]) -> jax.Array:
        # Horizontal speed of each foot while it is in contact, from the same
        # foot linear-velocity sensors `_cost_feet_clearance` reads. Playground's
        # version uses the base velocity instead, which penalises every stance
        # phase of walking.
        del info
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]  # (2, 3)
        return jp.sum(jp.linalg.norm(feet_vel[..., :2], axis=-1) * contact)

    def _obs_extra_args(self, data: Any) -> tuple:
        # Berkeley's `_get_obs(data, info, contact)` takes the foot-contact
        # flags; recompute them from the same sensors its reset/step use.
        contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self._feet_floor_found_sensor
        ])
        return (contact,)


ENV_CLASSES = {"go1": Go1JoystickERFI, "a1": A1JoystickERFI, "bh": BerkeleyHumanoidJoystickERFI}


def register() -> None:
    """Register the Go1 variant with Playground's locomotion suite. Idempotent.

    Afterwards `mujoco_playground._src.locomotion.load(ENV_NAME, ...)` works.
    The top-level `mujoco_playground.registry.load` only dispatches names that
    were present at import time, so prefer `erfi.load(...)` in our own code.
    """
    register_environment(ENV_NAME, Go1JoystickERFI, default_config)


def load(config: config_dict.ConfigDict | None = None, **config_overrides: Any):
    """Build the env for `config.robot` on `config.task` (no registry lookup needed)."""
    cfg = default_config() if config is None else config
    robot = cfg.get("robot", "go1")
    if robot not in ENV_CLASSES:
        raise ValueError(f"config.robot must be one of {ROBOTS}, got {robot!r}")
    return ENV_CLASSES[robot](config=cfg, config_overrides=config_overrides or None)


def domain_randomizer(task: str = "flat_terrain", robot: str = "go1"):
    """Playground's dynamics randomizer for the robot's task; the `dr` condition uses it.

    Go1 and A1 share Go1's function: it works by index (geom 0 = floor, body 1 =
    trunk, dofs 6: = the 12 joints), and both scenes satisfy that layout
    (verified in tests). The Berkeley Humanoid has its own function with the
    same index conventions but different terms (torso mass +-1 kg and rest-pose
    jitter instead of a CoM jitter); see docs/humanoid_design.md 2.1.
    """
    return pg_registry.get_domain_randomizer(playground_env_name(robot, task))
