"""Smoke tests for the environment catalogue.

Deliberately limited to metadata: these must stay fast and must not need the
2 GB Menagerie download, so nothing here builds a physics model. Tests that do
belong on the pod.
"""

from __future__ import annotations

import pytest

from rl_locomotion.envs import registry as reg


def test_lists_all_three_suites():
    infos = reg.list_envs()
    suites = {i.suite for i in infos}
    assert suites == set(reg.SUITES)
    assert len(infos) > 50


def test_locomotion_contains_expected_envs():
    names = {i.name for i in reg.list_envs("locomotion")}
    assert {"Go1JoystickFlatTerrain", "T1JoystickFlatTerrain"} <= names


@pytest.mark.parametrize(
    "env_name,platform,task",
    [
        ("Go1JoystickFlatTerrain", "go1", "joystick"),
        ("Go1Getup", "go1", "getup"),
        ("T1JoystickRoughTerrain", "t1", "joystick"),
        # dm_control_suite is flat — the module is the model.
        ("CartpoleBalance", "cartpole", "cartpole"),
    ],
)
def test_platform_and_task_resolution(env_name, platform, task):
    info = reg.env_info(env_name)
    assert (info.platform, info.task) == (platform, task)


def test_timing_fields_are_consistent():
    info = reg.env_info("Go1JoystickFlatTerrain")
    assert info.ctrl_dt > info.sim_dt
    assert info.sim_steps_per_ctrl_step == round(info.ctrl_dt / info.sim_dt)
    assert info.episode_seconds == pytest.approx(info.episode_length * info.ctrl_dt)


def test_go1_has_a_domain_randomizer():
    assert reg.env_info("Go1JoystickFlatTerrain").has_domain_randomizer


def test_unknown_env_raises():
    with pytest.raises(KeyError):
        reg.env_info("NotARobot")


def test_unknown_suite_raises():
    with pytest.raises(ValueError, match="unknown suite"):
        reg.list_envs("not_a_suite")


def test_models_group_envs_by_platform():
    go1 = next(m for m in reg.list_models("locomotion") if m.platform == "go1")
    assert go1.n_envs == len(go1.envs) >= 5
    assert any(x.endswith(".xml") for x in go1.xmls)


def test_barkour_ships_no_local_xml():
    """Not a bug: barkour loads its scene straight from Menagerie."""
    barkour = next(
        m for m in reg.list_models("locomotion") if m.platform == "barkour"
    )
    assert barkour.xmls == []


def test_envs_table_is_indexed_by_name():
    df = reg.envs_table("locomotion")
    assert df.index.name == "name"
    assert "Go1JoystickFlatTerrain" in df.index


def test_show_models_preserves_every_env_name():
    """The styled view exists because the plain table truncates; make sure the
    underlying data is complete regardless of how pandas chooses to print it."""
    df = reg.models_table("locomotion", sep="\n")
    listed = {n for cell in df["envs"] for n in cell.split("\n")}
    assert listed == {i.name for i in reg.list_envs("locomotion")}
