r"""Extended Random Force Injection (ERFI) on the Go1 / A1 joystick task.

Campanaro et al., "Learning and Deploying Robust Locomotion Policies with
Minimal Dynamics Randomization" (arXiv:2209.12878).

Actions stay reference joint positions — the perturbation is a torque added
on top of the actuator output, at the joint DOFs:

    tau_j = Kp(q*_j - q_j) - Kd qdot_j  +  tau_r,j  +  tau_o,j
            \_________ position actuator _________/    \___ qfrc_applied ___/

    RFI  tau_r ~ U(-r_lim, r_lim)   resampled every control step
    RAO  tau_o ~ U(-o_lim, o_lim)   sampled once per episode

Modes:
    "rfi"      tau_r only
    "rao"      tau_o only
    "erfi_c"   both, every episode
    "erfi_50"  each episode is RFI or RAO with probability 1/2 (paper default)

Observation (paper Sec. VI-B, the blind A1 setup), with H = history_len:

    gravity (3) | base linvel (3) | gyro (3) | joint-pos-error history (12H)
    | joint-vel history (12H) | previous action (12) | command (3)

H = 7 gives the paper's 192 dimensions; H = 1 is Playground's 48-dim state
reordered. The critic keeps Playground's privileged state so an asymmetric
critic remains available.

Robots. `cfg.robot` selects the model: "go1" (Playground's Go1JoystickFlatTerrain
/ RoughTerrain) or "a1" (our port of the same task to Menagerie's Unitree A1,
the robot of the paper's blind experiment; see envs/a1/). Both share every
name the task code uses, so the ERFI logic below is robot-agnostic.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground import registry as pg_registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion import register_environment
from mujoco_playground._src.locomotion.go1 import joystick

from rl_locomotion.envs.a1 import A1Joystick

ENV_NAME = "Go1JoystickERFI"
MODES = ("rfi", "rao", "erfi_c", "erfi_50")

# Playground scenes for the joystick task. The terrain is part of the config
# (`cfg.task`) rather than a constructor argument, so that a run's
# env_config.json fully determines the environment and evaluation can never
# silently rebuild a rough-terrain policy on flat ground.
#   flat_terrain   plane, floor friction 0.6, keyframe height 0.278 m (A1: 0.27)
#   rough_terrain  20 x 20 m heightfield, 5 cm peak-to-peak (std 1.45 cm),
#                  floor friction 1.0, keyframe height 0.35 m (A1: 0.34)
TASKS = ("flat_terrain", "rough_terrain")

# Robot models. Same rule: part of the config, recorded per run.
#   go1  Unitree Go1, 12.74 kg, limits 23.7 Nm hip/thigh, 35.55 Nm knee
#   a1   Unitree A1, 12.45 kg, limit 33.5 Nm on all joints (the paper's blind robot)
ROBOTS = ("go1", "a1")

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

# Layout of Playground's 48-dim state, used to slice the noisy joint readings.
# 0 to 2: base linear velocity, body frame
# 3 to 5: gyro
# 6 to 8: gravity vector, body frame
# 9 to 20: joint angles minus default pose
# 21 to 32: joint velocities
# 33 to 44: previous action
# 45 to 47: command
_JOINT_POS = slice(9, 21)
_JOINT_VEL = slice(21, 33)


def default_config() -> config_dict.ConfigDict:
    cfg = joystick.default_config()
    cfg.robot = "go1"
    cfg.task = "flat_terrain"
    # Paper: 7-step history of joint position errors and joint velocities.
    cfg.history_len = 7
    cfg.erfi = config_dict.create(
        enable=True,
        mode="erfi_50",
        # v2 recipe: append the episode's RAO offset (12) and the RFI flag (1)
        # to the critic's privileged state (123 -> 136). The policy input is
        # untouched. Lets the asymmetric critic explain return variance caused
        # by the hidden per-episode offset, which otherwise makes the advantage
        # of "start walking" noisy and stalls RAO/ERFI runs in the standing
        # optimum. v1 studies were trained with False.
        critic_sees_offset=False,
        # Nm. The paper uses 20 Nm on the 50 kg ANYmal C, about half a joint's
        # stance torque. Go1 and A1 are both ~12.5 kg, so the study uses 2.5 Nm
        # for both (set by the experiment config; 7.0 here is the historical
        # default that turned out too large, see the ERFI report).
        rfi_lim=7.0,
        rao_lim=7.0,
    )
    return cfg


def condition_config(name: str, **overrides: Any) -> config_dict.ConfigDict:
    """Default config for one of the study's training conditions."""
    if name not in CONDITIONS:
        raise ValueError(f"unknown condition {name!r}; choose from {list(CONDITIONS)}")
    cfg = default_config()
    cfg.erfi.enable = CONDITIONS[name]["erfi"]
    cfg.erfi.mode = CONDITIONS[name]["mode"]
    for k, v in overrides.items():
        cfg[k] = v
    if cfg.task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {cfg.task!r}")
    if cfg.robot not in ROBOTS:
        raise ValueError(f"robot must be one of {ROBOTS}, got {cfg.robot!r}")
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
    """ERFI torque perturbation + history observation, on top of a Go1-style joystick task.

    Must come first in the MRO; the base class is the robot-specific Joystick.
    """

    ROBOT: str = ""

    def __init__(self, task=None, config=None, config_overrides=None):
        cfg = default_config() if config is None else config
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
        # Playground's Joystick.__init__ overwrites naconmax/njmax for rough
        # terrain with its own (too small) values. Remember ours and restore
        # them afterwards; make_data reads them at reset time, so this is
        # enough. Never lower what Playground asked for.
        wanted_naconmax, wanted_njmax = cfg.naconmax, cfg.njmax
        super().__init__(task=cfg_task, config=cfg, config_overrides=config_overrides)
        self._config.naconmax = max(self._config.naconmax, wanted_naconmax)
        self._config.njmax = max(self._config.njmax, wanted_njmax)
        self._task = cfg_task
        mode = self._config.erfi.mode
        if mode not in MODES:
            raise ValueError(f"erfi.mode must be one of {MODES}, got {mode!r}")
        if self._config.history_len < 1:
            raise ValueError("history_len must be >= 1")
        # Joint DOFs sit after the 6 free-base DOFs.
        self._joint_dof_start = self.mjx_model.nv - self.mjx_model.nu

    @property
    def task(self) -> str:
        return self._task

    @property
    def robot(self) -> str:
        return self.ROBOT

    @property
    def floor_friction(self) -> float:
        """Sliding friction of the floor geom as trained (0.6 flat, 1.0 rough)."""
        return float(self.mj_model.geom_friction[self._floor_geom_id, 0])

    # ------------------------------------------------------------------ ERFI

    def _sample_erfi(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Per-episode: the RAO offset, and which channels this episode uses."""
        cfg = self._config.erfi
        # rng, k_off and k_mode are split here so the caller can use the leftover rng for other draws.
        # k_mode is used to decide whether to use RFI or RAO in the "erfi_50" mode.
        rng, k_off, k_mode = jax.random.split(rng, 3)

        tau_o = jax.random.uniform(
            k_off, (self.mjx_model.nu,), minval=-cfg.rao_lim, maxval=cfg.rao_lim
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

    # ----------------------------------------------------------- observation

    def _get_obs(self, data: Any, info: dict[str, Any]) -> dict[str, jax.Array]:
        obs = super()._get_obs(data, info)
        pg = obs["state"]
        joint_pos, joint_vel = pg[_JOINT_POS], pg[_JOINT_VEL]

        H = self._config.history_len
        if "joint_pos_hist" not in info:  # first call, from reset()
            # fill every slot with current value
            info["joint_pos_hist"] = jp.tile(joint_pos, (H, 1))
            info["joint_vel_hist"] = jp.tile(joint_vel, (H, 1))
        else:  # newest reading first
            # every slot moves one slot older
            info["joint_pos_hist"] = jp.roll(info["joint_pos_hist"], 1, axis=0).at[0].set(joint_pos)
            info["joint_vel_hist"] = jp.roll(info["joint_vel_hist"], 1, axis=0).at[0].set(joint_vel)

        state = jp.hstack([
            pg[6:9],  # gravity in body frame (orientation, yaw-free)
            pg[0:3],  # base linear velocity
            pg[3:6],  # gyro
            info["joint_pos_hist"].ravel(),
            info["joint_vel_hist"].ravel(),
            info["last_act"],
            info["command"],
        ])
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
        tau_o, use_rfi = self._sample_erfi(key)
        if not self._config.erfi.enable:
            tau_o, use_rfi = jp.zeros_like(tau_o), jp.zeros(())
        state.info["erfi_offset"] = tau_o
        state.info["erfi_use_rfi"] = use_rfi
        if self._config.erfi.get("critic_sees_offset", False):
            # The history is already filled with the current reading, so the
            # roll inside _get_obs leaves it unchanged; only the critic's extra
            # entries change.
            state = state.replace(obs=self._get_obs(state.data, state.info))
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg = self._config.erfi
        if cfg.enable:
            rng, key = jax.random.split(state.info["rng"])
            state.info["rng"] = rng
            tau_r = jax.random.uniform(
                key, (self.mjx_model.nu,), minval=-cfg.rfi_lim, maxval=cfg.rfi_lim
            )
            tau = state.info["erfi_offset"] + tau_r * state.info["erfi_use_rfi"]
            qfrc = jp.zeros(self.mjx_model.nv).at[self._joint_dof_start :].set(tau)
            state = state.replace(data=state.data.replace(qfrc_applied=qfrc))

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


ENV_CLASSES = {"go1": Go1JoystickERFI, "a1": A1JoystickERFI}


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
    """Playground's Go1 dynamics randomizer; the `dr` condition uses it.

    Playground registers the same function for both terrains. It works by
    index (geom 0 = floor, body 1 = trunk, dofs 6: = the 12 joints), and both
    the Go1 and A1 scenes satisfy that layout (verified in tests), so the same
    randomizer serves both robots.
    """
    del robot  # same function for go1 and a1
    name = {"flat_terrain": "Go1JoystickFlatTerrain", "rough_terrain": "Go1JoystickRoughTerrain"}[task]
    return pg_registry.get_domain_randomizer(name)
