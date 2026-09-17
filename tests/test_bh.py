"""CPU tests for the Berkeley Humanoid ERFI env and the generalised mixin.

Builds physics models on the JAX backend; JIT time dominates. Also pins the
Go1/A1 observations to a golden file recorded with the pre-generalisation
mixin (tests/golden/erfi_go1_a1_obs.npz), so a layout change on those robots
would show up here.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest

from rl_locomotion.envs import bh, erfi
from rl_locomotion.eval import perturb
from rl_locomotion.training import ppo

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden" / "erfi_go1_a1_obs.npz"
NU = 12


def _eval_terrain():
    """The scripts/ module, which is not an installed package."""
    import sys

    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import eval_terrain

    return eval_terrain


def _bh(condition="none", **overrides):
    return erfi.load(erfi.condition_config(condition, robot="bh", impl="jax", **overrides))


@pytest.fixture(scope="module")
def flat():
    return _bh("erfi_50", history_len=7)


@pytest.fixture(scope="module")
def flat_state(flat):
    return jax.jit(flat.reset)(jax.random.PRNGKey(0))


# ------------------------------------------------------------------ layout

def test_model_facts_match_the_design_doc(flat):
    m = flat.mj_model
    assert (m.nu, m.nv) == (12, 18)
    assert flat.total_mass == pytest.approx(16.06, abs=0.01)
    assert flat.spawn_height == pytest.approx(0.515)
    assert flat.floor_friction == pytest.approx(0.6)
    assert flat.n_substeps == 10
    assert [m.actuator(i).name for i in range(m.nu)] == list(bh.JOINT_NAMES)
    np.testing.assert_allclose(m.jnt_actfrcrange[1:, 1], bh.JOINT_ACTUATOR_FORCE_LIMIT)
    assert flat.layout.joint_pos_start == 12 and flat.layout.phase_dim == 4
    assert flat._joint_pos_slice == slice(12, 24) and flat._joint_vel_slice == slice(24, 36)
    assert not flat._config.push_config.enable  # study decision, design 4.6


def test_state_dimension_is_192_plus_phase(flat, flat_state):
    assert flat.observation_size == {"state": (196,), "privileged_state": (114,)}
    assert flat_state.obs["state"].shape == (196,)
    env1 = _bh("none", history_len=1)
    assert env1.observation_size["state"] == (52,)
    cfg = erfi.condition_config("rao", robot="bh", impl="jax")
    cfg.erfi.critic_sees_offset = True
    assert erfi.load(cfg).observation_size == {"state": (196,), "privileged_state": (127,)}


def test_state_is_playgrounds_reordered_with_history(flat, flat_state):
    """gravity | linvel | gyro | pos hist | vel hist | last_act | command | phase, from the 52-dim state.

    The noisy entries consumed the rng inside the base observation, so only the
    noise-free parts are compared against `info`.
    """
    s = flat_state.obs["state"]
    info = flat_state.info
    np.testing.assert_allclose(s[9 + 2 * NU * 7 : 9 + 2 * NU * 7 + NU], info["last_act"])
    np.testing.assert_allclose(s[9 + 2 * NU * 7 + NU : 9 + 2 * NU * 7 + NU + 3], info["command"])
    phase = jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])])
    np.testing.assert_allclose(s[-4:], phase, atol=1e-6)
    # At reset every history slot holds the same reading.
    hist = s[9 : 9 + NU * 7].reshape(7, NU)
    np.testing.assert_allclose(hist, jp.tile(hist[0], (7, 1)))


def test_privileged_state_starts_with_playgrounds(flat_state):
    # Playground's privileged state embeds its own 52-dim state first.
    assert flat_state.obs["privileged_state"].shape == (114,)


# ------------------------------------------------------------------ terrain

def test_rough_terrain_builds_and_rescales():
    rough = _bh("none", task="rough_terrain")
    assert rough.task == "rough_terrain"
    assert rough.floor_friction == pytest.approx(1.0)
    assert rough.spawn_height == pytest.approx(0.56)
    assert rough.terrain_amplitude == pytest.approx(0.05)
    assert rough._config.njmax >= 128
    small = _bh("none", task="rough_terrain", terrain_amplitude=0.02)
    assert small.terrain_amplitude == pytest.approx(0.02)
    state = jax.jit(rough.reset)(jax.random.PRNGKey(1))
    state = jax.jit(rough.step)(state, jp.zeros(NU))
    assert bool(jp.all(jp.isfinite(state.obs["state"])))


# ------------------------------------------------------------------ conditions

@pytest.mark.parametrize("condition", list(erfi.CONDITIONS))
def test_every_condition_steps_and_clears_qfrc(condition):
    env = _bh(condition, history_len=2)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = jax.jit(env.step)(state, 0.1 * jp.ones(NU))
    assert state.obs["state"].shape == (9 + 2 * NU * 2 + NU + 3 + 4,)
    assert bool(jp.all(jp.isfinite(state.obs["state"])))
    np.testing.assert_array_equal(state.data.qfrc_applied, jp.zeros(env.mjx_model.nv))
    if not erfi.CONDITIONS[condition]["erfi"]:
        np.testing.assert_array_equal(state.info["erfi_offset"], jp.zeros(NU))


def test_erfi_torque_lands_on_the_joint_dofs(flat):
    assert flat._joint_dof_start == 6  # six free-base DOFs, then the 12 joints


def test_low_base_terminates_but_the_spawn_pose_does_not(flat, flat_state):
    from mujoco import mjx

    assert flat._config.min_base_height == pytest.approx(0.5 * 0.515)
    term = jax.jit(flat._get_termination)
    assert not bool(term(flat_state.data))  # spawn at 0.515 m, upright
    sunk = flat_state.data.replace(qpos=flat_state.data.qpos.at[2].set(0.2))
    sunk = jax.jit(lambda d: mjx.forward(flat.mjx_model, d))(sunk)
    assert bool(term(sunk))  # torso still upright, but the base is below 0.2575 m
    off = _bh("none", min_base_height=0.0)
    assert not bool(jax.jit(off._get_termination)(sunk))  # 0 disables the extra rule


def test_feet_slip_uses_foot_velocity_not_base_velocity(flat, flat_state):
    d = flat_state.data
    contact = jp.ones(2, dtype=bool)
    # A moving base with feet at rest is walking, not slipping: the sensors read
    # the feet, so faking a base velocity changes nothing.
    fast_base = d.replace(qvel=d.qvel.at[0].set(1.0))
    assert float(flat._cost_feet_slip(fast_base, contact, {})) == pytest.approx(float(flat._cost_feet_slip(d, contact, {})))
    # Feet sliding at 0.3 m/s in x while in contact cost 2 x 0.3; no contact, no cost.
    sd = d.sensordata.at[flat._foot_linvel_sensor_adr].set(jp.array([[0.3, 0.0, 0.0]] * 2))
    sliding = d.replace(sensordata=sd)
    assert float(flat._cost_feet_slip(sliding, contact, {})) == pytest.approx(0.6, abs=1e-6)
    assert float(flat._cost_feet_slip(sliding, jp.zeros(2, dtype=bool), {})) == 0.0
    # Playground's version would have charged the base velocity instead.
    upstream = erfi.BerkeleyHumanoidJoystick._cost_feet_slip(flat, fast_base, contact, {})
    assert float(upstream) > 0.5


# ------------------------------------------------------------------ per-joint limits

def test_torque_limit_vector_is_applied_per_joint():
    lim = [0.0, 0.5, 1.0, 2.0, 3.0, 0.25] * 2
    cfg = erfi.condition_config("rao", robot="bh", impl="jax")
    erfi.set_torque_limits(cfg, lim, lim)
    env = erfi.load(cfg)
    assert env.rao_lim.shape == (NU,)
    offsets = jax.vmap(lambda k: env._sample_erfi(k)[0])(jax.random.split(jax.random.PRNGKey(0), 2000))
    hi = np.abs(np.asarray(offsets)).max(0)
    assert np.all(hi <= np.asarray(lim) + 1e-6)
    assert hi[0] == 0.0 and hi[6] == 0.0  # zero-limit joints get no offset
    assert np.all(hi[1:6] > 0.9 * np.asarray(lim[1:6]))  # and the others reach their own bound

    # RFI in `step` uses the same vector: with rfi_lim = 0 the erfi env moves like the
    # plain one, and with the provisional vector it does not.
    zero_cfg = erfi.condition_config("rfi", robot="bh", impl="jax")
    erfi.set_torque_limits(zero_cfg, [0.0] * NU, [0.0] * NU)
    zero_env, plain, rfi = erfi.load(zero_cfg), _bh("none"), _bh("rfi")
    a = jp.zeros(NU)
    s0 = jax.jit(zero_env.step)(jax.jit(zero_env.reset)(jax.random.PRNGKey(5)), a)
    s1 = jax.jit(plain.step)(jax.jit(plain.reset)(jax.random.PRNGKey(5)), a)
    s2 = jax.jit(rfi.step)(jax.jit(rfi.reset)(jax.random.PRNGKey(5)), a)
    np.testing.assert_allclose(s0.data.qpos, s1.data.qpos, atol=1e-6)
    assert not np.allclose(s2.data.qvel[6:], s1.data.qvel[6:])


def test_scalar_limit_still_accepted_and_wrong_length_rejected():
    cfg = erfi.condition_config("rfi", robot="bh", impl="jax")
    erfi.set_torque_limits(cfg, 1.5, 1.5)
    assert erfi.load(cfg).rfi_lim.shape == ()
    bad = erfi.condition_config("rfi", robot="bh", impl="jax")
    erfi.set_torque_limits(bad, [1.0] * 11, [1.0] * 11)
    with pytest.raises(ValueError, match="12 numbers"):
        erfi.load(bad)


def test_provisional_vector_is_ten_percent_of_the_joint_clamp():
    cfg = erfi.default_config("bh")
    np.testing.assert_allclose(cfg.erfi.rfi_lim, 0.1 * np.asarray(bh.JOINT_ACTUATOR_FORCE_LIMIT))
    assert erfi.default_config("go1").erfi.rfi_lim == 7.0


# ------------------------------------------------------------------ training plumbing

def test_train_spec_config_round_trips_through_json(tmp_path):
    lim = list(bh.PROVISIONAL_TORQUE_LIMIT)
    spec = ppo.TrainSpec(condition="erfi_50", robot="bh", task="rough_terrain", impl="jax",
                         rfi_lim=lim, rao_lim=lim, env_overrides={"history_target_error": True})
    cfg = ppo.env_config(spec)
    assert cfg.robot == "bh" and cfg.task == "rough_terrain"
    assert cfg.erfi.rfi_lim == lim and cfg.history_target_error is True
    (tmp_path / "env_config.json").write_text(json.dumps(cfg.to_dict(), default=str))
    loaded = ppo.load_env_config(tmp_path)
    assert list(loaded.erfi.rao_lim) == lim
    env = erfi.load(perturb.eval_env_config(loaded, impl="jax"))
    assert env.robot == "bh" and env.task == "rough_terrain"
    assert not env._config.erfi.enable and not env._config.push_config.enable
    assert env.observation_size["state"] == (196,)


def test_ppo_config_comes_from_playgrounds_berkeley_entry():
    p_bh = ppo.ppo_config(ppo.TrainSpec(condition="none", robot="bh"))
    p_go1 = ppo.ppo_config(ppo.TrainSpec(condition="none", robot="go1"))
    assert p_bh.entropy_cost == pytest.approx(0.005) and p_go1.entropy_cost == pytest.approx(0.01)
    assert p_bh.num_timesteps == 200_000_000 and p_bh.num_evals == 10
    assert tuple(p_bh.network_factory.policy_hidden_layer_sizes) == (512, 512)
    assert p_bh.num_resets_per_eval == 10  # design 4.4: same 5.6 s horizon as Go1


def test_domain_randomizer_is_the_robots_own():
    fn = erfi.domain_randomizer("rough_terrain", "bh")
    assert "berkeley_humanoid" in fn.__module__
    assert "go1" in erfi.domain_randomizer("flat_terrain", "go1").__module__
    env = _bh("dr")
    model, in_axes = fn(env.mjx_model, jax.random.split(jax.random.PRNGKey(0), 2))
    assert model.body_mass.shape == (2, env.mjx_model.nbody)
    assert erfi.uses_domain_randomization("dr")


# ------------------------------------------------------------------ protocol

def test_protocol_runs_batched_with_mass_scaled_levels_and_push_axes():
    env = erfi.load(perturb.eval_env_config(erfi.condition_config("erfi_50", robot="bh", task="rough_terrain"), impl="jax"))
    assert not env._config.erfi.enable  # RFI and RAO are off under the protocol
    assert not env._config.push_config.enable

    def zero_policy(obs, key):
        return jp.zeros(NU), None

    spec = perturb.EvalSpec(
        n_episodes=2, duration_s=0.2, push_start_s=0.0, push_duration_s=0.2,
        params=("payload_kg", "push_N_sagittal", "push_N_lateral"),
        level_fractions={"payload_kg": [0.0, 0.1], "push_N_sagittal": [0.32], "push_N_lateral": [0.32]},
        low_base_fraction=0.5, nominal_reset=True,
    )
    assert spec.levels_for("payload_kg", env) == pytest.approx([0.0, 1.6057], abs=1e-3)
    assert spec.levels_for("push_N_lateral", env) == pytest.approx([0.32 * 16.0568 * 9.81], abs=1e-2)
    df = perturb.evaluate_policy(env, zero_policy, spec, verbose=False, meta={"condition": "none", "seed": 0})
    assert list(df["param"]) == ["payload_kg", "payload_kg", "push_N_sagittal", "push_N_lateral"]
    assert set(["success_rate", "fall_rate", "low_base_rate", "progress_m", "tracking_rmse_alive", "robot_mass_kg"]) <= set(df.columns)
    assert (df["robot"] == "bh").all() and (df["task"] == "rough_terrain").all()
    assert df["robot_mass_kg"].iloc[0] == pytest.approx(16.06, abs=0.01)


def test_nominal_reset_starts_from_the_keyframe_at_rest(flat, flat_state):
    default_pose = flat.mj_model.keyframe("home").qpos[7:]
    # Playground's reset scales the joints by U(0.5, 1.5) and kicks the base.
    assert np.abs(np.asarray(flat_state.data.qpos[7:]) - default_pose).max() > 0.05
    assert np.abs(np.asarray(flat_state.data.qvel[:6])).max() > 0.0
    state = jax.jit(lambda s: perturb.nominal_start(flat, s))(flat_state)
    np.testing.assert_allclose(state.data.qpos[7:], default_pose, atol=1e-6)
    np.testing.assert_array_equal(state.data.qvel, jp.zeros(18))
    np.testing.assert_allclose(state.data.qpos[:7], flat_state.data.qpos[:7], atol=1e-6)  # base pose kept
    # Same pytree structure as the reset state (needed inside scan) and a refilled history:
    # every slot holds q - q_default plus at most 0.08 rad of sensor noise.
    assert set(state.info) == set(flat_state.info)
    hist = np.asarray(state.obs["state"])[9 : 9 + NU * 7].reshape(7, NU)
    np.testing.assert_allclose(hist, np.tile(hist[0], (7, 1)))
    assert np.abs(hist).max() < 0.1
    assert perturb.EvalSpec().nominal_reset is False  # Go1 protocol unchanged


def test_curriculum_config_builds_every_stage():
    """Each stage of the humanoid curriculum yields a valid env at its own relief."""
    import yaml

    cfg = yaml.safe_load((REPO / "configs/experiment/erfi_study_curr_bh.yaml").read_text())
    train, stages = dict(cfg["train"], impl="jax"), cfg["stages"]
    assert train["robot"] == "bh" and train["task"] == "rough_terrain"
    assert [s["terrain_amplitude"] for s in stages] == [0.0, 0.015, 0.03, 0.05]
    assert sum(s["num_timesteps"] for s in stages) == 200_000_000  # same budget as erfi_study_bh_rough
    for i, st in enumerate(stages):
        spec = ppo.TrainSpec(condition="erfi_50", seed=0, init_from="prev" if i else None,
                             terrain_amplitude=st["terrain_amplitude"],
                             num_timesteps=st["num_timesteps"], **train)
        env = erfi.load(ppo.env_config(spec))
        assert env.robot == "bh" and env.task == "rough_terrain"
        assert env.terrain_amplitude == pytest.approx(st["terrain_amplitude"])
        # friction and spawn height are constant across stages; only relief changes
        assert env.floor_friction == pytest.approx(1.0)
        assert env.spawn_height == pytest.approx(0.56)
        assert env.observation_size["state"] == (196,)
        assert len(env.rfi_lim) == NU


def test_terrain_suites_are_adjusted_for_the_humanoid():
    eval_terrain = _eval_terrain()

    for suite in eval_terrain.SUITES:
        go1 = eval_terrain.spec_for(suite, "go1", 50)
        bh = eval_terrain.spec_for(suite, "bh", 50)
        # Go1 is untouched by the humanoid entry. Combined suites carry a factorial
        # `grid` instead of one-at-a-time `params`, so theirs is empty.
        assert go1.params == tuple(eval_terrain.SUITES[suite].get("params", ()))
        assert go1.low_base_fraction is None and go1.nominal_reset is False
        # The humanoid always gets the kneeling flag and the keyframe start.
        assert bh.low_base_fraction == 0.5 and bh.nominal_reset is True
        assert "push_N" not in bh.params
    # The random push direction is replaced by the two named axes.
    prot = eval_terrain.spec_for("rough_bowl_protocol", "bh", 50)
    assert "push_N_sagittal" in prot.params and "push_N_lateral" in prot.params
    assert prot.params.index("push_N_sagittal") + 1 == prot.params.index("push_N_lateral")
    # Payload and push are fractions of this robot's own mass; friction stays absolute.
    assert set(prot.level_fractions) == {"payload_kg", "push_N_sagittal", "push_N_lateral"}
    assert "friction" in prot.levels and "payload_kg" not in prot.levels
    # Slope and relief grids straddle where this robot actually fails.
    assert eval_terrain.spec_for("bowl_slope", "bh", 50).levels["slope_deg"] == [0.0, 2.5, 5.0, 7.5, 10.0, 15.0]
    assert eval_terrain.spec_for("rough_relief", "bh", 50).levels["terrain_amplitude"] == [0.05, 0.075, 0.10, 0.125, 0.15]
    # Go1's fine sweep runs 10-26 deg; the humanoid falls by 10 deg, so its own
    # grid replaces those levels while the longer episode is kept.
    fine = eval_terrain.spec_for("bowl_slope_fine", "bh", 50)
    assert fine.levels["slope_deg"] == [0.0, 2.5, 5.0, 7.5, 10.0, 15.0] and fine.duration_s == 16.0


@pytest.mark.parametrize("shape", ["bowl", "rough_bowl"])
def test_humanoid_builds_and_steps_on_bowl_terrain(shape):
    cfg = erfi.condition_config("erfi_50", robot="bh", task="rough_terrain")
    cfg.terrain_shape, cfg.slope_deg = shape, 10.0
    env = erfi.load(perturb.eval_env_config(cfg, impl="jax"))
    assert float(env._config.hfield_elevation_cap) > 0  # needed to trace slope levels
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = jax.jit(env.step)(state, jp.zeros(NU))
    assert bool(jp.all(jp.isfinite(state.obs["state"])))
    # every swept level yields a traced model within the elevation cap
    prev = None
    for slope in [0.0, 2.5, 5.0, 7.5, 10.0, 15.0]:
        model = perturb.terrain_model(env, slope_deg=slope)
        top = float(jp.max(model.hfield_data))
        if prev is not None:
            assert top > prev  # a steeper bowl is a taller heightfield
        prev = top


def test_eval_spec_rejects_unknown_params():
    with pytest.raises(ValueError, match="unknown protocol parameter"):
        perturb.EvalSpec(params=("payload_kg", "push_N_diagonal"))
    assert perturb.base_param("push_N_lateral") == "push_N"
    assert perturb.EvalSpec().levels_for("push_N_sagittal") == perturb.PROTOCOL["push_N"][0]


# ------------------------------------------------------------------ Go1 / A1 unchanged

@pytest.mark.parametrize("robot", ["go1", "a1"])
def test_go1_and_a1_observations_match_the_pre_generalisation_golden(robot):
    golden = np.load(GOLDEN)
    cfg = erfi.condition_config("erfi_50", robot=robot, impl="jax", history_len=7)
    erfi.set_torque_limits(cfg, 2.5, 2.5)
    cfg.erfi.critic_sees_offset = True
    env = erfi.load(cfg)
    assert env.observation_size == {"state": (192,), "privileged_state": (136,)}
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.PRNGKey(7))
    acts = jax.random.uniform(jax.random.PRNGKey(11), (10, NU), minval=-1, maxval=1)
    states, privs = [np.asarray(state.obs["state"])], [np.asarray(state.obs["privileged_state"])]
    for a in acts:
        state = step(state, a)
        states.append(np.asarray(state.obs["state"]))
        privs.append(np.asarray(state.obs["privileged_state"]))
    np.testing.assert_allclose(np.stack(states), golden[f"{robot}_state"], atol=1e-5)
    np.testing.assert_allclose(np.stack(privs), golden[f"{robot}_privileged"], atol=1e-5)
    np.testing.assert_allclose(np.asarray(state.info["erfi_offset"]), golden[f"{robot}_offset"], atol=1e-6)
    np.testing.assert_allclose(np.asarray(state.data.qpos), golden[f"{robot}_qpos"], atol=1e-5)


def test_go1_defaults_and_protocol_levels_unchanged():
    cfg = erfi.default_config()
    assert cfg.robot == "go1" and cfg.erfi.rfi_lim == 7.0 and cfg.history_len == 7
    assert perturb.PROTOCOL["payload_kg"][0] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert perturb.PROTOCOL["push_N"][0] == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
    # The default protocol stays the paper's five sweeps: the push axes and the
    # terrain parameters are opt-in, never picked up by an existing config.
    assert perturb.EvalSpec().params == perturb.STANDARD_PARAMS
    assert perturb.STANDARD_PARAMS == ("payload_kg", "push_N", "friction", "gravity", "kp_scale")
    assert perturb.EvalSpec().low_base_fraction is None
    assert perturb.EvalSpec().nominal_reset is False
