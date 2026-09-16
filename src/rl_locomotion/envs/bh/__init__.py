"""Berkeley Humanoid joystick task, straight from MuJoCo Playground.

Unlike the A1 (a Go1 task on a different model), the humanoid has its own
Playground task with its own reward, gait clock and state layout. Nothing needs
to be re-implemented here; this package only names the base class the ERFI
mixin sits on and records the facts the mixin relies upon
(see docs/humanoid_design.md, Section 2):

    nu 12, nv 18 (six free-base DOFs first, as Go1)
    state 52 = linvel 3 | gyro 3 | gravity 3 | command 3 | joints 12 |
               joint_vel 12 | last_act 12 | phase 4
    privileged 114; ctrl_dt 0.02, sim_dt 0.002 (10 substeps); action_scale 0.5
    Kp 35 (ankle roll 1), Kd through dof_damping; jnt_actfrcrange per joint
    total mass 16.06 kg, torso 5.38 kg; floor friction 0.6 flat / 1.0 rough;
    home height 0.515 m flat / 0.56 m rough; feet-only collision
    built-in velocity pushes (`push_config`) on by default in Playground
"""

from mujoco_playground._src.locomotion.berkeley_humanoid import joystick as _joystick

BerkeleyHumanoidJoystick = _joystick.Joystick
default_config = _joystick.default_config

# Joint (= actuator) order of the model; the torque-limit vectors follow it.
JOINT_NAMES = (
    "LL_HR", "LL_HAA", "LL_HFE", "LL_KFE", "LL_FFE", "LL_FAA",
    "LR_HR", "LR_HAA", "LR_HFE", "LR_KFE", "LR_FFE", "LR_FAA",
)

# Joint-level clamp of the actuator force (`jnt_actfrcrange`, Nm). Injected
# ERFI torques bypass it (MJX clamps qfrc_actuator only), exactly as Go1's
# `actuator_forcerange` does not limit qfrc_applied.
JOINT_ACTUATOR_FORCE_LIMIT = (20.0, 20.0, 30.0, 30.0, 20.0, 5.0) * 2

# Provisional torque limit for smoke and timing runs only: 10 % of the joint
# clamp, the rule that gave Go1 its 2.5 Nm (10.5 % of 23.7 Nm). The study's
# vector is measured from a walking policy by scripts/measure_stance_torque.py.
PROVISIONAL_TORQUE_LIMIT = tuple(0.1 * x for x in JOINT_ACTUATOR_FORCE_LIMIT)

__all__ = [
    "BerkeleyHumanoidJoystick",
    "default_config",
    "JOINT_NAMES",
    "JOINT_ACTUATOR_FORCE_LIMIT",
    "PROVISIONAL_TORQUE_LIMIT",
]
