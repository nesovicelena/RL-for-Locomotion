"""CPU tests for the Unitree A1 port of the Go1 joystick task and its ERFI variant.

Builds physics models; about a minute of JIT time.
"""
from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest

from rl_locomotion.envs import erfi
from rl_locomotion.envs.a1 import A1Joystick
from rl_locomotion.eval import perturb
from rl_locomotion.training import ppo

ZERO = jp.zeros(12)


def _a1(condition="none", **overrides):
    return erfi.load(erfi.condition_config(condition, robot="a1", impl="jax", **overrides))


def _go1(condition="none", **overrides):
    return erfi.load(erfi.condition_config(condition, robot="go1", impl="jax", **overrides))


def _stand(env, n=50):
    """Zero action with a zero command for n control steps; returns the final state."""
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    state.info["command"] = jp.zeros(3)
    for _ in range(n):
        state = step(state, ZERO)
        state.info["command"] = jp.zeros(3)
    return state


def test_a1_model_matches_menagerie_numbers():
    env = _a1()
    m = env.mj_model
    assert m.nu == 12 and m.nv == 18 and m.nbody == 14
    assert float(m.body_mass.sum()) == pytest.approx(12.45, abs=0.02)
    assert float(m.body_mass[env._torso_body_id]) == pytest.approx(4.713)
    # A1: 33.5 Nm on every joint (Go1: 23.7 / 35.55)
    assert np.allclose(m.actuator_forcerange, [[-33.5, 33.5]] * 12)
    lo, hi = m.jnt_range[1:4].T
    assert np.allclose(lo, [-0.802851, -1.0472, -2.69653], atol=1e-5)
    assert np.allclose(hi, [0.802851, 4.18879, -0.916298], atol=1e-5)
    # same default pose as Go1, so action_scale / pose reward transfer
    assert np.allclose(np.array(env._default_pose), [0.1, 0.9, -1.8, -0.1, 0.9, -1.8] * 2)


def test_a1_shares_go1_layout_the_randomizer_relies_on():
    for env in (_a1(), _go1()):
        assert env._floor_geom_id == 0  # go1_randomize.FLOOR_GEOM_ID
        assert env._torso_body_id == 1  # go1_randomize.TORSO_BODY_ID
        assert env.mj_model.nv - env.mj_model.nu == 6  # dofs 6: are the 12 joints


def test_a1_interfaces_match_go1():
    a1, go1 = _a1("erfi_50"), _go1("erfi_50")
    assert a1.observation_size == go1.observation_size == {"state": (192,), "privileged_state": (123,)}
    assert a1.action_size == go1.action_size == 12
    assert a1.robot == "a1" and go1.robot == "go1"
    assert a1.dt == go1.dt and a1.n_substeps == go1.n_substeps


@pytest.mark.parametrize("task,height,min_up", [("flat_terrain", 0.27, 0.95), ("rough_terrain", 0.34, 0.8)])
def test_a1_stands_under_zero_action(task, height, min_up):
    # Playground's reset adds a random base velocity (+-0.5 m/s, rad/s); on the
    # heightfield that can leave the trunk tilted after 1 s. Go1 tilts the same
    # amount on the same seed, so the rough threshold is deliberately loose.
    env = _a1(task=task)
    state0 = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert float(state0.data.qpos[2]) == pytest.approx(height, abs=0.01)
    state = _stand(env)
    assert float(state.done) == 0.0
    assert float(env.get_upvector(state.data)[2]) > min_up
    # settles a little under the PD gains, but stays standing
    assert 0.20 < float(state.data.qpos[2]) < 0.36
    contacts = [bool(state.data.sensordata[env.mj_model.sensor_adr[s]] > 0) for s in env._feet_floor_found_sensor]
    # on the heightfield a tilted trunk can leave one foot hovering
    assert sum(contacts) >= (4 if task == "flat_terrain" else 3)


def test_a1_scenes_have_expected_floors():
    flat, rough = _a1(task="flat_terrain"), _a1(task="rough_terrain")
    assert int(flat.mj_model.geom_type[flat._floor_geom_id]) == 0  # plane
    assert int(rough.mj_model.geom_type[rough._floor_geom_id]) == 1  # hfield
    assert flat.floor_friction == pytest.approx(0.6)
    assert rough.floor_friction == pytest.approx(1.0)
    # same heightfield as Go1's rough scene
    go1_rough = _go1(task="rough_terrain")
    assert np.allclose(rough.mj_model.hfield_size, go1_rough.mj_model.hfield_size)
    assert np.allclose(rough.mj_model.hfield_data, go1_rough.mj_model.hfield_data)


@pytest.mark.parametrize("condition", list(erfi.CONDITIONS))
def test_a1_every_condition_steps(condition):
    env = _a1(condition)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    for _ in range(5):
        state = step(state, ZERO)
    assert state.obs["state"].shape == (192,)
    assert bool(jp.all(state.data.qfrc_applied == 0)), "perturbation torque leaked out of step"


def test_robot_is_enforced():
    cfg = erfi.condition_config("none", robot="a1")
    with pytest.raises(ValueError):
        erfi.Go1JoystickERFI(config=cfg)
    with pytest.raises(ValueError):
        erfi.condition_config("none", robot="anymal_c")
    with pytest.raises(ValueError):
        ppo.TrainSpec(condition="none", robot="anymal_c")


def test_a1_train_spec_and_eval_config():
    spec = ppo.TrainSpec(condition="erfi_50", robot="a1", task="rough_terrain", rfi_lim=2.5, rao_lim=2.5)
    cfg = ppo.env_config(spec)
    assert (cfg.robot, cfg.task, cfg.erfi.rfi_lim) == ("a1", "rough_terrain", 2.5)
    ecfg = perturb.eval_env_config(cfg, impl="jax")
    env = erfi.load(ecfg)
    assert isinstance(env, erfi.A1JoystickERFI)
    assert env.task == "rough_terrain" and not env._config.erfi.enable
    assert perturb.nominal_value(env, "friction") == pytest.approx(1.0)


def test_a1_protocol_runs():
    env = _a1(task="flat_terrain")
    env = erfi.load(perturb.eval_env_config(env._config, impl="jax"))
    spec = perturb.EvalSpec(n_episodes=2, duration_s=0.4, params=("payload_kg", "kp_scale"),
                            levels={"payload_kg": [0.0, 3.0], "kp_scale": [1.0, 0.5]})
    df = perturb.evaluate_policy(env, lambda o, k: (ZERO, None), spec, verbose=False,
                                 meta={"condition": "none", "seed": 0})
    assert len(df) == 4 and set(df["task"]) == {"flat_terrain"}
    assert float(perturb.perturbed_model(env, "payload_kg", 3.0).body_mass[env._torso_body_id]) == pytest.approx(4.713 + 3.0)


@pytest.mark.parametrize("shape", ["bowl", "rough_bowl"])
def test_a1_builds_and_steps_on_bowl_terrain(shape):
    cfg = erfi.condition_config("erfi_50", robot="a1", task="rough_terrain")
    cfg.terrain_shape, cfg.slope_deg = shape, 10.0
    env = erfi.load(perturb.eval_env_config(cfg, impl="jax"))
    assert float(env._config.hfield_elevation_cap) == 8.0   # needed to trace slope levels
    assert env.spawn_height == pytest.approx(0.34)          # A1's rough spawn, not Go1's
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = jax.jit(env.step)(state, ZERO)
    assert bool(jp.all(jp.isfinite(state.obs["state"])))
    # the whole fine sweep stays inside the elevation cap and rises with slope
    prev = None
    for slope in (10.0, 18.0, 26.0):
        peak = float(np.asarray(perturb.perturbed_model(env, "slope_deg", slope).hfield_data).max())
        assert peak * 8.0 <= 8.0 + 1e-6
        if prev is not None:
            assert peak > prev
        prev = peak


def test_a1_combined_grid_matches_the_go1_levels():
    """A1 is within 2% of Go1's mass with the same floor frictions, so the suites
    need no per-robot adjustment; this pins that, since a drift either way would
    silently make the two robots' numbers incomparable."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("eval_terrain", Path("scripts/eval_terrain.py"))
    et = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(et)

    a1 = _a1(task="rough_terrain")
    go1 = _go1(task="rough_terrain")
    assert perturb.robot_mass(a1) == pytest.approx(perturb.robot_mass(go1), rel=0.05)
    assert a1.floor_friction == pytest.approx(go1.floor_friction)
    assert _a1(task="flat_terrain").floor_friction == pytest.approx(_go1(task="flat_terrain").floor_friction)
    assert a1.observation_size["state"] == go1.observation_size["state"]
    assert "a1" not in et.ROBOT_ADJUST
    for suite in ("combined_relief", "combined_bowl"):
        assert et.grid_for(suite, "a1", a1) == et.grid_for(suite, "go1", go1) == et.COMBINED_GRID
    for suite in et.SUITES:
        assert et.spec_for(suite, "a1", 50) == et.spec_for(suite, "go1", 50)


def test_a1_study_configs_cover_v1_v3_and_the_curriculum():
    import yaml
    from pathlib import Path

    def load(name):
        return yaml.safe_load(Path(f"configs/experiment/{name}.yaml").read_text())

    plain = {
        "erfi_study_a1": ("flat_terrain", False),
        "erfi_study_a1_rough": ("rough_terrain", False),
        "erfi_study_a1_v3": ("flat_terrain", True),
        "erfi_study_a1_v3_rough": ("rough_terrain", True),
    }
    for name, (task, v3) in plain.items():
        c = load(name)
        assert c["out"] == f"{name}_l2.5"
        assert c["conditions"] == ["none", "dr", "rfi", "rao", "erfi_c", "erfi_50"] and c["seeds"] == [0, 1, 2]
        assert c["train"]["robot"] == "a1" and c["train"]["task"] == task
        assert c["train"]["rfi_lim"] == 2.5 and c["train"]["rao_lim"] == 2.5
        # v3 is exactly v1 plus the q* - q history, so that the pair isolates it
        assert bool(c["train"].get("env_overrides", {}).get("history_target_error")) is v3
        # the rough studies centre the friction sweep on the rough scene's 1.0
        assert ("friction" in c["eval"].get("levels", {})) is (task == "rough_terrain")
        spec = ppo.TrainSpec(condition="rao", seed=0, **c["train"])
        assert erfi.load(ppo.env_config(spec)).robot == "a1"

    curr = load("erfi_study_curr_a1_v3")
    assert curr["out"] == "erfi_study_curr_a1_v3_l2.5"
    assert curr["train"]["robot"] == "a1" and curr["train"]["task"] == "rough_terrain"
    assert curr["train"]["env_overrides"]["history_target_error"] is True
    assert [s["terrain_amplitude"] for s in curr["stages"]] == [0.0, 0.015, 0.03, 0.05]
    assert sum(s["num_timesteps"] for s in curr["stages"]) == 300_000_000
    # the Go1 curriculum it mirrors, stage for stage
    go1_curr = load("erfi_study_curr_v3")
    assert curr["stages"] == go1_curr["stages"] and curr["eval"] == go1_curr["eval"]
