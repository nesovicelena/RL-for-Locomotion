"""Boston Dynamics Spot joystick task: Playground's own task, plus a rough-terrain scene.

Unlike A1 (a Go1 task on another model), Spot has its own Playground task with
its own reward, PD gains (Kp 300, Kd 1 on the joints plus kv 20 in the
actuator) and action scale (0.3), all validated by Playground on this robot. The
study keeps that task whole -- the same choice made for the Berkeley Humanoid --
and the ERFI mixin builds the paper's 192-dim policy input on top of it
(envs/erfi.py, `LAYOUTS["spot"]`).

What Playground does not ship is a rough-terrain scene for Spot. This module adds
one (xmls/spot_scene_rough_terrain.xml): Playground's Go1 heightfield, the terrain
of every other rough study here, under Playground's own Spot model and contact
sensors, with the keyframe raised 0.07 m for clearance. The flat task is
Playground's unchanged.

Facts the mixin relies on (checked in tests/test_spot.py):
    nu 12, nv 18; floor geom 0, torso body 1 -> Go1's domain randomizer applies
    state: gyro 0:3 | gravity 3:6 | joints-default 6:18 | qpos error history |
           feet pos 12 | last_act 12 | command 3   (no base linvel, no joint vel)
    total mass 50.34 kg, torso 32.86 kg; keyframe 0.46 m flat / 0.53 m rough
    termination: gravity z < 0.85, i.e. tilt beyond ~32 deg (Go1: past horizontal)
    motor targets clipped to the actuator ctrlrange (Go1 does not clip)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import mujoco
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import go1_constants as go1_consts
from mujoco_playground._src.locomotion.spot import base as spot_base
from mujoco_playground._src.locomotion.spot import joystick as spot_joystick
from mujoco_playground._src.locomotion.spot import spot_constants as consts

XML_DIR = epath.Path(__file__).parent / "xmls"
ROUGH_XML = XML_DIR / "spot_scene_rough_terrain.xml"
TASKS = ("flat_terrain", "rough_terrain")

# Joint (= actuator) order of the model, for per-joint torque-limit reports.
JOINT_NAMES = ("fl_hx", "fl_hy", "fl_kn", "fr_hx", "fr_hy", "fr_kn",
               "hl_hx", "hl_hy", "hl_kn", "hr_hx", "hr_hy", "hr_kn")


def get_assets() -> Dict[str, bytes]:
    """Playground's Spot assets plus our scene and the Go1 heightfield, keyed by basename."""
    assets = spot_base.get_assets()
    mjx_env.update_assets(assets, XML_DIR, "*.xml")
    mjx_env.update_assets(assets, go1_consts.ROOT_PATH / "xmls" / "assets")
    return assets


class SpotJoystick(spot_joystick.Joystick):
    """Playground's Spot joystick task with a rough_terrain variant."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = spot_joystick.default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; choose from {TASKS}")
        if task == "flat_terrain":
            super().__init__(task=task, config=config, config_overrides=config_overrides)
            return

        # Same contact budget every rough scene here uses; never lower the config's.
        config.naconmax = max(config.naconmax, 8 * 8192)
        config.njmax = max(config.njmax, 12 + 48)

        # Replicates SpotEnv.__init__ with our XML and asset dict; Joystick.__init__
        # hard-codes Playground's flat scene through consts.task_to_xml.
        mjx_env.MjxEnv.__init__(self, config, config_overrides)
        self._model_assets = get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(ROUGH_XML.read_text(), assets=self._model_assets)
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.dof_damping[6:] = self._config.Kd
        self._mj_model.actuator_gainprm[:, 0] = self._config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -self._config.Kp
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = ROUGH_XML.as_posix()
        self._feet_floor_found_sensor = [
            self._mj_model.sensor(f"{geom}_floor_found").id for geom in consts.FEET_GEOMS
        ]
        self._imu_site_id = self._mj_model.site("imu").id

        # Joystick.__init__ tail: default pose, ids, sensors, posture weights, pushes.
        self._post_init()
        self._pert_func = (
            self._maybe_apply_perturbation if config.pert_config.enable else lambda state, _: state
        )
