"""CPU tests for the Boston Dynamics Spot ERFI environment (Playground's Spot task + our rough scene).

Builds physics models; about a minute of JIT time.
"""
from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest

from rl_locomotion.envs import erfi
from rl_locomotion.envs.terrain import height_map

ZERO = jp.zeros(12)


def _spot(condition="none", **overrides):
    return erfi.load(erfi.condition_config(condition, robot="spot", impl="jax", **overrides))


def _stand(env, n=50):
    """Zero action with a zero command for n control steps; returns the final state."""
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    state.info["command"] = jp.zeros(3)
    for _ in range(n):
        state = step(state, ZERO)
        state.info["command"] = jp.zeros(3)
    return state


def test_spot_model_facts():
    env = _spot()
    m = env.mj_model
    assert m.nu == 12 and m.nv == 18 and m.nbody == 14
    assert env.total_mass == pytest.approx(50.34, abs=0.01)
    assert float(m.body_mass[env._torso_body_id]) == pytest.approx(32.86, abs=0.01)
    # Playground's Spot gains through the config; kv 20 stays in the actuator.
    assert np.allclose(m.actuator_gainprm[:, 0], 300.0) and np.allclose(m.dof_damping[6:], 1.0)
    assert np.allclose(m.actuator_biasprm[:, 2], -20.0)
    assert not m.actuator_forcelimited.any()  # no torque clamp on this model
    assert np.allclose(np.array(env._default_pose), [0.0, 1.04, -1.8] * 4)


def test_spot_shares_go1_layout_the_randomizer_relies_on():
    env = _spot()
    assert env._floor_geom_id == 0  # go1_randomize.FLOOR_GEOM_ID
    assert env._torso_body_id == 1  # go1_randomize.TORSO_BODY_ID
    assert env.mj_model.nv - env.mj_model.nu == 6
    fn = erfi.domain_randomizer("flat_terrain", "spot")
    model, _ = fn(env.mjx_model, jax.random.split(jax.random.PRNGKey(0), 3))
    assert len(set(np.round(np.asarray(model.geom_friction[:, 0, 0]), 4))) == 3


def test_spot_policy_input_is_the_papers_192():
    env = _spot("erfi_50")
    assert env.observation_size["state"] == (192,) and env.action_size == 12
    assert env.robot == "spot"


def test_spot_base_readings_come_from_the_simulator():
    """Spot's state has no base linvel or joint velocities; the mixin reads them from data, with Go1's noise."""
    env = _spot()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = jax.jit(env.step)(state, 0.2 * jp.ones(12))
    obs = state.obs["state"]
    assert bool(jp.all(jp.abs(obs[0:3] - env.get_gravity(state.data)) <= 0.03 + 1e-6))
    assert bool(jp.all(jp.abs(obs[3:6] - env.get_local_linvel(state.data)) <= 0.1 + 1e-6))
    assert bool(jp.all(jp.abs(obs[6:9] - env.get_gyro(state.data)) <= 0.1 + 1e-6))
    assert bool(jp.all(jp.abs(state.info["joint_vel_hist"][0] - state.data.qvel[6:]) <= 1.5 + 1e-6))


def test_spot_v3_history_uses_the_clipped_target():
    env = _spot(history_target_error=True)
    state = env.reset(jax.random.PRNGKey(0))
    a = 5.0 * jp.ones(12)  # far past the actuator range on purpose
    state = env.step(state, a)
    default = jp.asarray(env._default_pose)
    clipped = jp.clip(default + a * env._config.action_scale, env._lowers, env._uppers)
    assert bool(jp.allclose(state.info["motor_targets"], clipped, atol=1e-6))
    err = state.info["joint_pos_hist"][0] - (clipped - state.data.qpos[7:])
    assert bool(jp.all(jp.abs(err) <= 0.05 + 1e-6))  # joint-position noise only


@pytest.mark.parametrize("task,height,friction", [("flat_terrain", 0.46, 0.6), ("rough_terrain", 0.53, 1.0)])
def test_spot_scenes(task, height, friction):
    env = _spot(task=task)
    assert env.spawn_height == pytest.approx(height) and env.floor_friction == pytest.approx(friction)
    if task == "rough_terrain":
        hm = height_map(env.mj_model)
        assert hm.shape == (256, 256) and hm.radius_x == 10.0 and hm.peak_to_peak == pytest.approx(0.05)
        assert env._config.naconmax == 8 * 8192 and env._config.njmax == 128
    state = _stand(env)
    assert float(state.done) == 0.0
    assert float(env.get_gravity(state.data)[2]) > 0.95
    assert 0.35 < float(state.data.qpos[2]) < 0.6


def test_spot_fall_criterion_is_the_tasks_own():
    """Spot terminates at gravity z < 0.85 (~32 deg tilt), not past horizontal like Go1."""
    from mujoco.mjx._src import math as mjxm

    env = _spot()
    for deg, want in ((40.0, 1.0), (20.0, 0.0)):
        state = env.reset(jax.random.PRNGKey(0))
        quat = mjxm.axis_angle_to_quat(jp.array([0.0, 1.0, 0.0]), jp.radians(deg))
        state = state.replace(data=state.data.replace(qpos=state.data.qpos.at[3:7].set(quat)))
        assert float(env.step(state, ZERO).done) == want


def test_spot_erfi_torque_lands_on_the_joints():
    env = _spot("rfi")
    state = env.reset(jax.random.PRNGKey(0))
    state = env.step(state, ZERO)
    # cleared after the step, so read the offset/flag plumbing instead
    assert state.info["erfi_use_rfi"] == 1.0 and state.info["erfi_offset"].shape == (12,)
    assert bool(jp.all(state.data.qfrc_applied == 0.0))


def test_spot_protocol_levels_are_go1_fractions():
    import importlib.util
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "scripts" / "eval_terrain.py"
    spec = importlib.util.spec_from_file_location("eval_terrain", src)
    et = importlib.util.module_from_spec(spec); spec.loader.exec_module(et)
    env = _spot()
    s = et.spec_for("rough_bowl_protocol", "spot", 50)
    payload, push = s.levels_for("payload_kg", env), s.levels_for("push_N", env)
    assert payload[-1] == pytest.approx(6.0 / 12.743 * 50.34, abs=0.05)      # 23.7 kg
    assert push[-1] == pytest.approx(40.0 / (12.743 * 9.81) * 50.34 * 9.81, abs=0.5)  # 158 N
    grid = et.grid_for("combined_bowl", "spot", env)
    assert len(grid["payload_kg"]) == 3 and len(grid["push_N"]) == 3 and grid["friction"] == [1.0, 0.6, 0.3]
    assert grid["payload_kg"][1] == pytest.approx(3.0 / 12.743 * 50.34, abs=0.05)  # the 3 kg point
    assert grid["push_N"][1] == pytest.approx(20.0 / (12.743 * 9.81) * 50.34 * 9.81, abs=0.5)  # the 20 N point
