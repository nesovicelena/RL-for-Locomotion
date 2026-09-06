r"""Extended Random Force Injection (ERFI) on the Go1 joystick task.

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

ENV_NAME = "Go1JoystickERFI"
MODES = ("rfi", "rao", "erfi_c", "erfi_50")

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
    # Paper: 7-step history of joint position errors and joint velocities.
    cfg.history_len = 7
    cfg.erfi = config_dict.create(
        enable=True,
        mode="erfi_50",
        # Nm. The paper uses 20-40 Nm on ANYmal C (limit ~80 Nm), i.e. roughly
        # 25-50% of peak torque. Go1's limits are 23.7 Nm (hip/thigh) and
        # 35.55 Nm (knee), so the equivalent band is ~6-12 Nm.
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
    return cfg


def uses_domain_randomization(name: str) -> bool:
    return CONDITIONS[name]["randomize"]


class Go1JoystickERFI(joystick.Joystick):
    """Go1 joystick with ERFI torque perturbations and a history observation."""

    def __init__(self, task="flat_terrain", config=None, config_overrides=None):
        super().__init__(
            task=task,
            config=default_config() if config is None else config,
            config_overrides=config_overrides,
        )
        mode = self._config.erfi.mode
        if mode not in MODES:
            raise ValueError(f"erfi.mode must be one of {MODES}, got {mode!r}")
        if self._config.history_len < 1:
            raise ValueError("history_len must be >= 1")
        # Joint DOFs sit after the 6 free-base DOFs.
        self._joint_dof_start = self.mjx_model.nv - self.mjx_model.nu

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
        return {"state": state, "privileged_state": obs["privileged_state"]}

    # ----------------------------------------------------------- reset/step

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key = jax.random.split(rng)
        state = super().reset(rng)
        tau_o, use_rfi = self._sample_erfi(key)
        if not self._config.erfi.enable:
            tau_o, use_rfi = jp.zeros_like(tau_o), jp.zeros(())
        state.info["erfi_offset"] = tau_o
        state.info["erfi_use_rfi"] = use_rfi
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


def register() -> None:
    """Register with Playground's locomotion suite. Idempotent.

    Afterwards `mujoco_playground._src.locomotion.load(ENV_NAME, ...)` works.
    The top-level `mujoco_playground.registry.load` only dispatches names that
    were present at import time, so prefer `erfi.load(...)` in our own code.
    """
    register_environment(ENV_NAME, Go1JoystickERFI, default_config)


def load(config: config_dict.ConfigDict | None = None, **config_overrides: Any) -> Go1JoystickERFI:
    """Build the env directly (no registry lookup needed)."""
    return Go1JoystickERFI(config=config, config_overrides=config_overrides or None)


def domain_randomizer():
    """Playground's Go1 dynamics randomizer; the `dr` condition uses it."""
    return pg_registry.get_domain_randomizer("Go1JoystickFlatTerrain")
