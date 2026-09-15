"""Unitree A1 joystick task: Playground's Go1 task on the A1 model.

Why this is a thin subclass and not a new environment: Menagerie's A1 and Go1
share the kinematic tree and every name Playground's Go1 code refers to
(bodies `trunk`, `FR_hip`, ...; joints `FR_hip_joint`, ...; actuators; foot
geoms and sites `FR`, `FL`, `RR`, `RL`; the `imu` site; all sensors). Only the
numbers differ (Section: xmls/a1_mjx_feetonly.xml). So the reward terms,
observation code, termination and command sampling of `go1.joystick.Joystick`
apply verbatim, and only the model construction is overridden.

Differences from Go1 that matter for training:
    trunk mass 4.71 kg (Go1 5.20), total ~12.5 kg (Go1 12.74)
    thigh/calf 0.200 m (Go1 0.213), foot radius 0.020 m (Go1 0.023)
    actuator limit 33.5 Nm on all joints (Go1 23.7 hip/thigh, 35.55 knee)
    joint ranges: abduction +-0.803, hip -1.047..4.189, knee -2.697..-0.916
    keyframe height 0.27 m flat / 0.34 m rough (Go1 0.278 / 0.35)
Kp/Kd, action_scale, reward weights and max_foot_height (0.10 m) are inherited
from the Go1 config. The paper's A1 used Kp=15, Kd=1; Playground's Go1 uses
Kp=35, Kd=0.5. Both are exposed through the config (`Kp`, `Kd`).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

from etils import epath
import mujoco
from mujoco import mjx
from ml_collections import config_dict
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import go1_constants as go1_consts
from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick

XML_DIR = epath.Path(__file__).parent / "xmls"
TASK_TO_XML = {
    "flat_terrain": XML_DIR / "a1_scene_flat_terrain.xml",
    "rough_terrain": XML_DIR / "a1_scene_rough_terrain.xml",
}


def get_assets() -> Dict[str, bytes]:
    """Everything the A1 scenes reference, keyed by basename (MuJoCo's VFS convention).

    A1 meshes come from Menagerie; the rough-terrain heightfield is Playground's
    own Go1 asset, so the rough scene is the same terrain as Go1JoystickRoughTerrain.
    """
    assets: Dict[str, bytes] = {}
    mjx_env.update_assets(assets, XML_DIR, "*.xml")
    mjx_env.update_assets(assets, mjx_env.MENAGERIE_PATH / "unitree_a1" / "assets")
    mjx_env.update_assets(assets, go1_consts.ROOT_PATH / "xmls" / "assets")
    return assets


class A1Joystick(go1_joystick.Joystick):
    """Go1 joystick task on the Unitree A1 model."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = go1_joystick.default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if task not in TASK_TO_XML:
            raise ValueError(f"unknown task {task!r}; choose from {list(TASK_TO_XML)}")
        if task.startswith("rough"):
            # Same contact budget Playground gives Go1 on the heightfield.
            config.naconmax = max(config.naconmax, 8 * 8192)
            config.njmax = max(config.njmax, 12 + 48)

        # Replicates Go1Env.__init__ (base.py) with our XML and asset dict. We
        # deliberately skip Joystick.__init__ and Go1Env.__init__, which hard-code
        # the Go1 XML path and the Go1 asset directory.
        mjx_env.MjxEnv.__init__(self, config, config_overrides)
        xml_path = TASK_TO_XML[task]
        self._model_assets = get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(xml_path.read_text(), assets=self._model_assets)
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.opt.ccd_iterations = 20

        self._mj_model.dof_damping[6:] = self._config.Kd
        self._mj_model.actuator_gainprm[:, 0] = self._config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -self._config.Kp

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = xml_path.as_posix()
        self._imu_site_id = self._mj_model.site("imu").id
        self._feet_floor_found_sensor = [
            self._mj_model.sensor(f"{geom}_floor_found").id for geom in go1_consts.FEET_GEOMS
        ]

        # Joystick._post_init: default pose, joint limits, body/site/geom ids, command config.
        self._post_init()
