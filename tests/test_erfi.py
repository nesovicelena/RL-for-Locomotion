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


def test_training_wrapper_resets_history_and_offset_on_done():
    """Through the training wrappers, a fall must leave a clean history and a new RAO offset.

    Playground's default auto-reset restores data/obs to a cached first state
    but keeps `info`, which left the previous episode's fallen-pose readings in
    the history for history_len - 1 steps. Training uses full_reset=True.
    """
    import functools
    from mujoco_playground import wrapper

    env = _env("rao")
    H = env._config.history_len
    wrapped = functools.partial(wrapper.wrap_for_brax_training, full_reset=True)(
        env, episode_length=1000, action_repeat=1
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    state = jax.jit(wrapped.reset)(keys)
    step = jax.jit(wrapped.step)
    act = jp.zeros((2, 12))
    for _ in range(3):
        state = step(state, act)
    offset_before = state.info["erfi_offset"][0]
    # flip env 0 upside down so it terminates on the next step, with the joints
    # far from the default pose so a stale history is visible
    qpos = state.data.qpos.at[0, 3:7].set(jp.array([0.0, 1.0, 0.0, 0.0])).at[0, 7:].set(0.0)
    state = step(state.replace(data=state.data.replace(qpos=qpos)), act)
    assert float(state.done[0]) == 1.0 and float(state.done[1]) == 0.0
    state = step(state, act)  # first step of the new episode
    hist = state.obs["state"][0, 9 : 9 + 12 * H].reshape(H, 12)
    assert bool(jp.all(jp.abs(hist - hist[0]) < 0.1)), "history still holds the previous episode"
    assert not bool(jp.allclose(state.info["erfi_offset"][0], offset_before))


def test_history_target_error_is_q_star_minus_q():
    cfg = erfi.condition_config("none", impl="jax")
    cfg.history_target_error = True
    cfg.noise_config.level = 0.0  # exact comparison
    env = erfi.load(cfg)
    default = env._default_pose
    scale = cfg.action_scale
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))
    assert state.obs["state"].shape == (192,)
    # same pytree structure after reset and step (needed by scan / auto-reset)
    tree = jax.tree_util.tree_structure
    act = jp.linspace(-0.5, 0.5, 12)
    nxt = step(state, act)
    assert tree(state) == tree(nxt)
    # at reset the target is the keyframe pose: error = q_default - q
    hist0 = state.obs["state"][9 : 9 + 12 * 7].reshape(7, 12)
    assert jp.allclose(hist0[0], default - state.data.qpos[7:], atol=1e-5)
    # after a step the newest slot is (q_default + a * scale) - q for the action just applied
    hist1 = nxt.obs["state"][9 : 9 + 12 * 7].reshape(7, 12)
    assert jp.allclose(hist1[0], default + act * scale - nxt.data.qpos[7:], atol=1e-5)
    assert jp.allclose(hist1[1], hist0[0])
    # default recipe unchanged: q - q_default
    plain_cfg = erfi.condition_config("none", impl="jax")
    plain_cfg.noise_config.level = 0.0
    s = jax.jit(erfi.load(plain_cfg).reset)(jax.random.PRNGKey(0))
    assert jp.allclose(s.obs["state"][9:21], s.data.qpos[7:] - default, atol=1e-5)


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


# ------------------------------------------------------------------ bowl terrains

def test_bowl_terrain_geometry_and_standing():
    import numpy as np
    from rl_locomotion.envs.terrain import height_map

    env = _env("none", task="rough_terrain", terrain_shape="bowl", slope_deg=15.0)
    hm = height_map(env.mj_model)
    # flat disc of radius 1 m, then tan(15 deg) per metre; 3 % short because of the 7.8 cm grid
    assert hm.sample(0.0, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert hm.sample(0.8, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert hm.sample(3.0, 0.0) == pytest.approx(np.tan(np.radians(15.0)) * 2.0, rel=0.05)
    assert hm.sample(0.0, -3.0) == pytest.approx(hm.sample(3.0, 0.0), rel=0.05)  # radially symmetric
    state = _run(env, 25)
    assert float(env.get_upvector(state.data)[2]) > 0.95 and float(state.done) == 0.0


def test_rough_bowl_adds_relief_on_top_of_slope():
    import numpy as np
    from rl_locomotion.envs.terrain import height_map

    env = _env("none", task="rough_terrain", terrain_shape="rough_bowl", slope_deg=15.0, terrain_amplitude=0.05)
    hm = height_map(env.mj_model)
    xs = np.linspace(-hm.radius_x, hm.radius_x, hm.shape[1]); row = hm.heights[hm.shape[0] // 2]
    disc = row[np.abs(xs) < 0.9]
    assert 0.03 < disc.max() - disc.min() < 0.06          # rocky relief survives on the flat disc
    ramp = (np.abs(xs) > 1.5) & (np.abs(xs) < 3.5)
    slope = np.degrees(np.arctan(np.polyfit(np.abs(xs[ramp]), row[ramp], 1)[0]))
    assert slope == pytest.approx(15.0, abs=1.0)
    with pytest.raises(ValueError):
        _env("none", task="rough_terrain", terrain_shape="pyramid")


# ------------------------------------------------------- terrain protocol params

def test_terrain_params_are_traced_heightfield_data():
    import numpy as np
    from rl_locomotion.envs.terrain import height_map

    cfg = perturb.eval_env_config(erfi.condition_config("none", task="rough_terrain", terrain_shape="rough_bowl",
                                                        slope_deg=10.0), impl="jax")
    assert cfg.hfield_elevation_cap == 8.0
    env = erfi.load(cfg)
    assert float(env.mj_model.hfield_size[0, 2]) == 8.0          # fixed elevation
    assert env.terrain_amplitude == pytest.approx(0.05)          # relief read from config, not the rim
    assert perturb.nominal_value(env, "slope_deg") == 10.0
    assert perturb.nominal_value(env, "terrain_amplitude") == pytest.approx(0.05)
    # a steeper bowl only changes hfield_data; sizes (static in MJX) are untouched
    m30 = perturb.perturbed_model(env, "slope_deg", 30.0)
    assert np.allclose(np.asarray(m30.hfield_size), np.asarray(env.mjx_model.hfield_size))
    d0, d30 = np.asarray(env.mjx_model.hfield_data), np.asarray(m30.hfield_data)
    assert d30.max() > d0.max() * 2                                # 30 deg rim ~5.2 m vs 10 deg ~1.6 m
    assert d30.max() * 8.0 < 8.0 + 1e-6                            # within the cap
    with pytest.raises(ValueError):
        perturb.perturbed_model(env, "slope_deg", 45.0)          # rim 9 m > cap
    # relief sweep keeps the slope: heights at 3 m differ by at most the relief change
    a10 = np.asarray(perturb.perturbed_model(env, "terrain_amplitude", 0.10).hfield_data) * 8.0
    hm0 = height_map(env.mj_model)
    idx = hm0.shape[0] // 2, int((3.0 + hm0.radius_x) / (2 * hm0.radius_x) * (hm0.shape[1] - 1))
    assert abs(a10.reshape(hm0.shape)[idx] - hm0.heights[idx]) < 0.06
    # flat-terrain env cannot take terrain params
    with pytest.raises(ValueError):
        perturb.perturbed_model(erfi.load(perturb.eval_env_config(erfi.condition_config("none"), impl="jax")), "slope_deg", 10.0)


def test_terrain_protocol_runs_batched():
    cfg = perturb.eval_env_config(erfi.condition_config("none", task="rough_terrain", terrain_shape="bowl"), impl="jax")
    env = erfi.load(cfg)
    spec = perturb.EvalSpec(n_episodes=2, duration_s=0.4, params=("slope_deg", "terrain_amplitude"),
                            levels={"slope_deg": [0.0, 20.0], "terrain_amplitude": [0.05]})
    df = perturb.evaluate_policy(env, lambda o, k: (ZERO, None), spec, verbose=False, meta={"condition": "none", "seed": 0})
    assert list(df["param"]) == ["slope_deg", "slope_deg", "terrain_amplitude"]
    assert (df["success_rate"] == 0).all() and df["progress_m"].abs().max() < 0.5


def test_curriculum_v3_config():
    import yaml
    from rl_locomotion.training import ppo

    c = yaml.safe_load(Path("configs/experiment/erfi_study_curr_v3.yaml").read_text())
    assert c["out"] == "erfi_study_curr_v3_l2.5"
    assert sum(s["num_timesteps"] for s in c["stages"]) == 300_000_000
    assert [s["terrain_amplitude"] for s in c["stages"]] == [0.0, 0.015, 0.03, 0.05]
    spec = ppo.TrainSpec(condition="rao", seed=0, terrain_amplitude=0.015, num_timesteps=1, **c["train"])
    cfg = ppo.env_config(spec)
    assert cfg.history_target_error is True and cfg.task == "rough_terrain" and cfg.terrain_shape == "playground"


def test_grid_points_is_full_factorial_in_a_stable_order():
    pts = perturb.grid_points({"a": [1.0, 2.0], "b": [3.0, 4.0, 5.0]})
    assert len(pts) == 6
    assert pts[0] == {"a": 1.0, "b": 3.0} and pts[-1] == {"a": 2.0, "b": 5.0}
    assert len({perturb.combo_label(p) for p in pts}) == 6
    assert perturb.combo_label({"payload_kg": 3.0, "friction": 0.5}) == "payload_kg=3|friction=0.5"


def test_combined_model_applies_every_parameter_at_once():
    import numpy as np

    env = erfi.load(perturb.eval_env_config(erfi.condition_config("none"), impl="jax"))
    torso, floor = env._torso_body_id, env._floor_geom_id
    m0 = env.mjx_model
    combo = {"payload_kg": 6.0, "friction": 0.3, "push_N": 40.0, "gravity": -15.0, "kp_scale": 0.75}
    m = perturb.combined_model(env, combo)
    # every model-side parameter is in effect simultaneously
    assert float(m.body_mass[torso]) == pytest.approx(float(m0.body_mass[torso]) + 6.0)
    assert float(m.geom_friction[floor, 0]) == pytest.approx(0.3)
    assert float(m.opt.gravity[2]) == pytest.approx(-15.0)
    assert np.allclose(np.asarray(m.actuator_gainprm[:, 0]), np.asarray(m0.actuator_gainprm[:, 0]) * 0.75)
    # the push is not a model field; it is read back for the rollout
    assert perturb.push_of(combo) == (40.0, 0)
    assert perturb.push_of({"payload_kg": 1.0}) == (0.0, 0)
    assert perturb.push_of({"push_N_lateral": 5.0}) == (5.0, 2)
    # composing is order-independent and matches applying each one alone
    m_rev = perturb.combined_model(env, dict(reversed(list(combo.items()))))
    assert float(m_rev.body_mass[torso]) == pytest.approx(float(m.body_mass[torso]))
    assert float(m_rev.geom_friction[floor, 0]) == pytest.approx(0.3)
    # a single-entry combo equals the one-at-a-time model
    assert float(perturb.combined_model(env, {"payload_kg": 6.0}).body_mass[torso]) == pytest.approx(
        float(perturb.perturbed_model(env, "payload_kg", 6.0).body_mass[torso]))


def test_evaluate_policy_grid_rows_and_nominal_flag():
    env = erfi.load(perturb.eval_env_config(erfi.condition_config("none"), impl="jax"))
    grid = {"payload_kg": [0.0, 6.0], "friction": [0.6, 0.3], "push_N": [0.0, 40.0]}
    spec = perturb.EvalSpec(n_episodes=2, duration_s=0.4)
    df = perturb.evaluate_policy_grid(env, lambda o, k: (ZERO, None), grid, spec, verbose=False,
                                      meta={"condition": "none", "seed": 0})
    assert len(df) == 8
    assert (df["param"] == "combined").all()
    assert list(df["combo_id"]) == list(range(8))
    assert set(df.columns) >= {"lvl_payload_kg", "lvl_friction", "lvl_push_N", "combo", "n_perturbed"}
    # friction 0.6 is flat ground's training value, so the all-nominal corner is the first row
    assert df.loc[0, "nominal"] == 1.0 and df.loc[0, "n_perturbed"] == 0
    assert (df.loc[1:, "nominal"] == 0.0).all()
    assert df["n_perturbed"].max() == 3
    # a zero policy never reaches 2.5 m
    assert (df["success_rate"] == 0).all()


def test_new_terrain_suites_are_wired():
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_terrain", Path("scripts/eval_terrain.py"))
    et = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(et)

    # the fine slope sweeps run longer episodes than the 8 s protocol
    for name in ("bowl_slope_fine", "rough_bowl_slope_fine"):
        assert et.SUITES[name]["levels"]["slope_deg"][0] == 10.0
        assert et.SUITES[name]["levels"]["slope_deg"][-1] == 26.0
        assert et.spec_for(name, "go1", 50).duration_s == 16.0
    assert et.spec_for("bowl_slope", "go1", 50).duration_s == 8.0

    # the combined suites carry a 27-point grid and no one-at-a-time params
    for name in ("combined_relief", "combined_bowl"):
        cfg = et.SUITES[name]
        assert "params" not in cfg and len(perturb.grid_points(cfg["grid"])) == 27
        assert et.spec_for(name, "go1", 50).params == ()
    assert et.SUITES["combined_bowl"]["env"]["slope_deg"] == 10.0
    assert et.SUITES["combined_relief"]["env"]["terrain_shape"] == "playground"

    # the quadruped grid passes through untouched
    env = erfi.load(perturb.eval_env_config(erfi.condition_config("none"), impl="jax"))
    assert et.grid_for("combined_bowl", "go1", env) == et.COMBINED_GRID
    # the humanoid's grid is rescaled to its own mass and uses one named push axis
    bh_grid = et.grid_for("combined_bowl", "bh", env)
    assert "push_N_sagittal" in bh_grid and "push_N" not in bh_grid
    assert len(bh_grid["payload_kg"]) == 3 and bh_grid["payload_kg"][0] == 0.0
