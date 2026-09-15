"""CPU tests for the ERFI environment and the perturbation protocol.

These build physics models, so they need the Menagerie assets and take about a
minute of JIT time. Run them before pushing anything the pod will train with.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jp
import pytest

from rl_locomotion.envs import erfi
from rl_locomotion.eval import perturb

ZERO = jp.zeros(12)


@pytest.fixture(scope="module")
def rng():
    return jax.random.PRNGKey(0)


def _env(condition: str, **overrides):
    return erfi.load(erfi.condition_config(condition, impl="jax", **overrides))


def _run(env, n=10, act=ZERO):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    for _ in range(n):
        state = step(state, act)
    return state


def test_state_dimension_matches_paper():
    env = _env("erfi_50")
    assert env.observation_size["state"] == (192,)
    assert _env("none", history_len=1).observation_size["state"] == (48,)


def test_history_rolls_newest_first():
    env = _env("none")
    state = _run(env, 3)
    prev_newest = state.info["joint_pos_hist"][0]
    state = jax.jit(env.step)(state, ZERO)
    assert jp.allclose(state.info["joint_pos_hist"][1], prev_newest)


@pytest.mark.parametrize("condition", list(erfi.CONDITIONS))
def test_every_condition_steps(condition):
    env = _env(condition)
    state = _run(env, 5)
    assert state.obs["state"].shape == (192,)
    assert bool(jp.all(state.data.qfrc_applied == 0)), "perturbation torque leaked out of step"


def test_erfi_changes_the_dynamics():
    base = _run(_env("none"), 25).data.qpos[7:]
    for cond in ("rfi", "rao", "erfi_c"):
        pert = _run(_env(cond), 25).data.qpos[7:]
        assert float(jp.abs(pert - base).max()) > 0.05, cond


def test_rao_offset_is_redrawn_on_done():
    env = _env("rao")
    state = _run(env, 2)
    before = state.info["erfi_offset"]
    flipped = state.data.replace(qpos=state.data.qpos.at[3:7].set(jp.array([0.0, 1.0, 0.0, 0.0])))
    state = jax.jit(env.step)(state.replace(data=flipped), ZERO)
    assert float(state.done) == 1.0
    assert not bool(jp.allclose(state.info["erfi_offset"], before))


def test_perturbed_models():
    env = _env("none")
    m = env.mjx_model
    t = env._torso_body_id
    assert float(perturb.perturbed_model(env, "payload_kg", 3.0).body_mass[t]) == pytest.approx(float(m.body_mass[t]) + 3.0)
    assert float(perturb.perturbed_model(env, "friction", 0.2).geom_friction[env._floor_geom_id, 0]) == pytest.approx(0.2)
    assert float(perturb.perturbed_model(env, "gravity", -2.0).opt.gravity[2]) == pytest.approx(-2.0)
    kp = perturb.perturbed_model(env, "kp_scale", 0.5)
    assert float(kp.actuator_gainprm[0, 0]) == pytest.approx(17.5)
    assert float(kp.actuator_biasprm[0, 1]) == pytest.approx(-17.5)


def test_protocol_runs_batched():
    env = erfi.load(perturb.eval_env_config(erfi.condition_config("none"), impl="jax"))

    def zero_policy(obs, key):
        return jp.zeros(12), None

    spec = perturb.EvalSpec(n_episodes=2, duration_s=0.4, params=("payload_kg", "push_N"),
                            levels={"payload_kg": [0.0, 2.0], "push_N": [0.0, 20.0]})
    df = perturb.evaluate_policy(env, zero_policy, spec, verbose=False, meta={"condition": "none", "seed": 0})
    assert len(df) == 4
    assert set(df["param"]) == {"payload_kg", "push_N"}
    assert df["success_rate"].between(0, 1).all()
    # standing still for 0.4 s cannot cover 2.5 m
    assert (df["success_rate"] == 0).all()


# ------------------------------------------------------------- rough terrain

def test_task_is_part_of_the_config():
    flat = erfi.condition_config("none")
    rough = erfi.condition_config("none", task="rough_terrain")
    assert flat.task == "flat_terrain" and rough.task == "rough_terrain"
    # contact limits recorded in the config match what Playground uses for the scene
    assert (flat.naconmax, flat.njmax) == (4 * 8192, 40)
    assert (rough.naconmax, rough.njmax) == (8 * 8192, 128)
    with pytest.raises(ValueError):
        erfi.condition_config("none", task="stairs")


def test_rough_terrain_builds_a_different_scene():
    flat = _env("none")
    rough = _env("none", task="rough_terrain")
    assert flat.task == "flat_terrain" and rough.task == "rough_terrain"
    # heightfield floor vs plane, and the two scenes' floor friction differ
    assert int(rough.mj_model.geom_type[rough._floor_geom_id]) == 1  # mjGEOM_HFIELD
    assert int(flat.mj_model.geom_type[flat._floor_geom_id]) == 0  # mjGEOM_PLANE
    assert flat.floor_friction == pytest.approx(0.6)
    assert rough.floor_friction == pytest.approx(1.0)
    # same robot, same interfaces
    assert rough.observation_size == flat.observation_size
    assert rough.action_size == flat.action_size


def test_task_argument_must_agree_with_config():
    cfg = erfi.condition_config("none", task="rough_terrain")
    with pytest.raises(ValueError):
        erfi.Go1JoystickERFI(task="flat_terrain", config=cfg)


def test_rough_terrain_steps_and_perturbs():
    env = _env("erfi_50", task="rough_terrain")
    state = _run(env, 5)
    assert state.obs["state"].shape == (192,)
    assert bool(jp.all(state.data.qfrc_applied == 0))


def test_eval_config_keeps_the_terrain():
    cfg = perturb.eval_env_config(erfi.condition_config("rfi", task="rough_terrain"), impl="jax")
    assert cfg.task == "rough_terrain"
    assert not cfg.erfi.enable
    env = erfi.load(cfg)
    assert env.task == "rough_terrain"
    assert perturb.nominal_value(env, "friction") == pytest.approx(1.0)
    assert perturb.nominal_value(_env("none"), "friction") == pytest.approx(0.6)


# ------------------------------------------------------------------- v2 recipe

def test_v1_default_is_unchanged():
    env = _env("erfi_50")
    assert not env._config.erfi.critic_sees_offset
    assert env.observation_size == {"state": (192,), "privileged_state": (123,)}


def test_v2_critic_sees_offset_but_actor_does_not():
    cfg = erfi.condition_config("rao", impl="jax")
    cfg.erfi.critic_sees_offset = True
    env = erfi.load(cfg)
    assert env.observation_size == {"state": (192,), "privileged_state": (136,)}
    state = jax.jit(env.reset)(jax.random.PRNGKey(3))
    # the critic's extra entries equal the episode's offset, from the very first observation
    assert jp.allclose(state.obs["privileged_state"][123:135], state.info["erfi_offset"])
    assert float(state.obs["privileged_state"][135]) == 0.0  # rao: RFI channel off
    assert bool(jp.any(state.info["erfi_offset"] != 0))
    # ... and stay in sync after a step
    state = jax.jit(env.step)(state, ZERO)
    assert jp.allclose(state.obs["privileged_state"][123:135], state.info["erfi_offset"])
    assert state.obs["state"].shape == (192,)


def test_v2_zero_extras_when_erfi_disabled():
    cfg = erfi.condition_config("none", impl="jax")
    cfg.erfi.critic_sees_offset = True
    env = erfi.load(cfg)
    state = _run(env, 3)
    assert bool(jp.all(state.obs["privileged_state"][123:] == 0))


def test_v2_config_applies_overrides():
    import yaml
    from pathlib import Path
    from rl_locomotion.training import ppo

    c = yaml.safe_load(Path("configs/experiment/erfi_study_v2_rough.yaml").read_text())
    spec = ppo.TrainSpec(condition="erfi_c", seed=0, **c["train"])
    cfg = ppo.env_config(spec)
    assert cfg.reward_config.tracking_sigma == 0.1
    assert cfg.erfi.critic_sees_offset is True
    assert (cfg.robot, cfg.task) == ("go1", "rough_terrain")
    assert c["out"] == "erfi_study_v2_rough_l2.5"


# ------------------------------------------------------------ terrain curriculum

def test_terrain_amplitude_rescales_the_heightfield():
    from rl_locomotion.envs.terrain import height_map

    for amp in (0.0, 0.015, 0.05):
        env = _env("none", task="rough_terrain", terrain_amplitude=amp)
        assert env.terrain_amplitude == pytest.approx(amp)
        assert height_map(env.mj_model).peak_to_peak == pytest.approx(amp, abs=1e-6)
        assert float(env.mjx_model.hfield_size[0, 2]) == pytest.approx(amp)
    # flat terrain ignores it
    assert _env("none", task="flat_terrain", terrain_amplitude=0.03).terrain_amplitude == 0.0
    with pytest.raises(ValueError):
        _env("none", task="rough_terrain", terrain_amplitude=-0.01)


def test_curriculum_spec_and_config():
    import yaml
    from pathlib import Path
    from rl_locomotion.training import ppo

    c = yaml.safe_load(Path("configs/experiment/erfi_study_curr.yaml").read_text())
    amps = [s["terrain_amplitude"] for s in c["stages"]]
    assert amps == [0.0, 0.015, 0.03, 0.05]
    assert sum(s["num_timesteps"] for s in c["stages"]) == 200_000_000
    spec = ppo.TrainSpec(condition="rao", seed=1, terrain_amplitude=0.015, init_from="/nonexistent",
                         num_timesteps=1, **c["train"])
    cfg = ppo.env_config(spec)
    assert cfg.terrain_amplitude == 0.015 and cfg.task == "rough_terrain"
    with pytest.raises(FileNotFoundError):
        ppo.train(spec, Path("/tmp/erfi_curr_should_not_train"))


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "experiments/erfi_study_l2.5/none/seed0/params_final").exists(),
                    reason="needs a finished v1 run to restore from")
def test_restore_from_run_trains(tmp_path):
    """Tiny PPO run initialised from a v1 policy: exercises the init_from path end to end."""
    from rl_locomotion.training import ppo

    src = Path(__file__).resolve().parents[1] / "experiments/erfi_study_l2.5/none/seed0"
    spec = ppo.TrainSpec(
        condition="none", seed=0, task="rough_terrain", terrain_amplitude=0.0, init_from=str(src),
        impl="jax", num_timesteps=512, num_evals=2,
        ppo_overrides=dict(num_envs=4, batch_size=4, num_minibatches=2, unroll_length=8,
                           episode_length=32, num_resets_per_eval=1, num_eval_envs=4),
    )
    out = ppo.train(spec, tmp_path / "run")
    assert (tmp_path / "run" / "params_final").exists()
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["init_from"] == str(src)
    # a restored (walking) policy scores far above a fresh random one even on a tiny eval
    assert out["curve"][0]["reward"] > 0.2
