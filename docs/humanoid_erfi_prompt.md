# Prompt: extend the ERFI study from Go1 to a Playground humanoid

Paste everything below the line into a fresh Claude Code session opened in
this repository.

---

## Context

Repo: RL-for-Locomotion (MuJoCo Playground 0.14.2 + Brax 0.14.2 PPO, MJX
Warp for training, JAX backend for evaluation). We replicate Campanaro et al.,
"Learning and Deploying Robust Locomotion Policies with Minimal Dynamics
Randomization" (arXiv:2209.12878, `./2209.12878v2.pdf`) on the Unitree Go1.
Six training conditions (`none`, `dr`, `rfi`, `rao`, `erfi_c`, `erfi_50`),
three seeds, torque limit 2.5 Nm, 200 M steps. Everything that matters is in:

- `src/rl_locomotion/envs/erfi.py`: `_ERFIMixin` on top of Playground's Go1
  joystick task. Adds RFI/RAO torques through `qfrc_applied` at the joint
  DOFs, a 7-step history observation (192-dim), `cfg.task` (flat/rough),
  `cfg.robot` (go1/a1), `cfg.terrain_amplitude`, `cfg.history_target_error`,
  `cfg.erfi.critic_sees_offset`.
- `src/rl_locomotion/envs/a1/`: the pattern for adding a robot (a thin
  subclass that only swaps the model; the mixin is robot-agnostic as long as
  the task shares Go1's names and layout).
- `src/rl_locomotion/training/ppo.py`: `TrainSpec`, one self-describing run
  directory per (condition, seed), `load_policy`, `load_env_config`.
- `src/rl_locomotion/eval/perturb.py`: the paper's Section VII protocol
  (0.5 m/s, 8 s, success = no fall and >= 2.5 m; payload, push, friction,
  gravity, Kp; 50 episodes per level).
- `scripts/train.py`, `scripts/eval.py`, `runpod/run_all_studies.sh`,
  `configs/experiment/erfi_study*.yaml`.
- `docs/eval_design.md`: the evaluation design for the quadruped recipes, and
  the fairness rules (seed treatment, walking-only subsets, what is and is not
  from the paper).
- `docs/report/erfi_studija_sr.tex`, `docs/report/make_erfi_figures.py`:
  report format and figure style (fixed colour per condition, marker as the
  secondary encoding).
- `tests/test_erfi.py`, `tests/test_a1.py`: CPU tests, about four minutes.

Task: design, then implement, the same study on one Playground humanoid.
Work in the order below and stop for my review after step 1 and after step 3.
Do not run training or full evaluations locally; they run on the RunPod GPU.
Do not modify anything under `experiments/`. Do not change the behaviour of
the Go1/A1 code paths or any existing config; every existing test must still
pass.

## Facts already verified in the installed Playground (do not re-derive, do re-check if versions changed)

Three humanoid joystick tasks share Go1's template. All use PD position
actuators with `motor_targets = default_pose + action * action_scale`, keep
the `info` keys `last_act`, `last_last_act`, `command`, `motor_targets`, and
place the joints after exactly six free-base DOFs, so `nv - nu == 6` and the
mixin's `qfrc_applied` injection at `nv - nu` works unchanged.

| env | nu | nv | state dim | privileged dim | total mass | Kp range per joint | ctrl_dt / sim_dt | action_scale | randomizer | rough variant |
|---|---|---|---|---|---|---|---|---|---|---|
| BerkeleyHumanoidJoystickFlatTerrain | 12 | 18 | 52 | 114 | 16.1 kg | 1-35 | 0.02 / 0.002 | 0.5 | yes | yes |
| T1JoystickFlatTerrain | 23 | 29 | 85 | 180 | 31.6 kg | 10-50 | 0.02 / 0.002 | 1.0 | yes | yes |
| G1JoystickFlatTerrain | 29 | 35 | 103 | 216 | 33.3 kg | 2-75 | 0.02 / 0.002 | 0.5 | yes | yes |
| Go1JoystickFlatTerrain (reference) | 12 | 18 | 48 | 123 | 12.7 kg | 35 | 0.02 / 0.004 | 0.5 | yes | yes |

Differences from Go1 that the code must handle:

1. **State layout.** Humanoids: `linvel(3) | gyro(3) | gravity(3) | command(3) |
   joint_angles - default_pose(nu) | joint_vel(nu) | last_act(nu) | phase`.
   Go1: `linvel | gyro | gravity | joints | joint_vel | last_act | command`.
   The mixin's `_JOINT_POS = slice(9, 21)` and `_JOINT_VEL = slice(21, 33)`
   are Go1-only. The joint slices must become `slice(12, 12 + nu)` and
   `slice(12 + nu, 12 + 2 nu)` for humanoids, derived from `nu`, not typed.
2. **Phase term.** The humanoid state ends with a gait-clock `phase` (cos/sin
   of a per-foot phase, driven by `gait_freq` in the config). It is not in the
   paper's state, but it is part of Playground's task and its reward, so keep
   it in the reordered state and count it in the new state dimension. Report
   the resulting state size per robot in `docs/humanoid_design.md`.
3. **No actuator force limits.** `actuator_forcerange` is zero (unlimited) on
   all three humanoids. The Go1 study scaled the torque limit from the
   23.7 Nm motor limit ("about 10 % of a joint's limit, roughly half a
   stance torque"). For humanoids, derive the limit per joint instead: measure
   the actuator force during a nominal stand of Playground's own pretrained
   policy or a zero-action stand for 2 s (`data.actuator_force`), take the
   per-joint RMS, and set `rfi_lim`/`rao_lim` as a fraction of that vector
   (start at 0.5, the paper's "half stance torque"). The limit therefore
   becomes a per-joint array in the config, with the scalar form still
   accepted for Go1/A1.
4. **Termination.** Berkeley Humanoid and T1 terminate on gravity z < 0 plus
   NaN guards; G1 additionally terminates on self-contact between feet and
   shins. The protocol's `fallen` flag reads `state.done`, so no change, but
   note G1's extra rule in the design.
5. **Substeps.** sim_dt is 0.002, so ten physics substeps per control step
   instead of five. RFI is still resampled once per control step and held
   across substeps; state this deviation from the paper as the Go1 report did.
6. **Kp is per joint** (1 to 75). The protocol's `kp_scale` multiplies
   `actuator_gainprm[:, 0]` and `actuator_biasprm[:, 1]`, which is already
   per-actuator, so the sweep works; the domain randomisers of the three
   humanoids should be read to confirm they use the same index conventions
   the Go1 one does (floor geom 0, torso body 1, dofs 6:) before reusing the
   `dr` condition machinery.
7. **Mass scaling of the protocol.** Payload and push levels were scaled to
   Go1 by mass ratio to ANYmal C (12.7 / 50). Rescale to each humanoid's mass
   (16.1 / 31.6 / 33.3 kg): payload up to about 20 % of body mass, push up to
   about 150 N x (mass / 50).
8. **Command and reward.** The humanoid tasks have their own command ranges,
   reward scales and `tracking_sigma`. Do not port the v2 `tracking_sigma 0.1`
   blindly; read each task's default and compute the standing-still tracking
   reward at a 0.5 m/s command before deciding.

## Step 1: design document (stop for review)

Write `docs/humanoid_design.md` that:

a. Picks the robot with a one-paragraph justification. Default
   recommendation: **Berkeley Humanoid** first (12 actuators, same nu/nv as
   Go1, smallest mass, cheapest training), G1 as the stretch target because it
   has the paper-like scale (33 kg, 29 DOF) and a self-contact termination.
b. Lists, from the table above and from the chosen task's `joystick.py`,
   exactly what the mixin needs to generalise: joint slices, state size,
   per-joint torque limit, protocol mass scaling, nominal friction and spawn
   height, any info keys the humanoid task adds or lacks.
c. Proposes the study: which conditions (all six if the randomizer checks
   out), seeds (3 to match Go1; argue for 5 if wall-clock allows), budget in
   steps (measure one 20 M-step run on the pod first and extrapolate; Go1 took
   5.5 min per 200 M, humanoids will be several times slower with ten
   substeps and more contacts), flat first, rough second.
d. Lists the parameters that need tuning and how each will be chosen, with
   the measurement that decides it, not a guess: torque-limit fraction
   (stance-torque measurement, then a short 50 M sweep at 0.25 / 0.5 / 1.0 on
   `erfi_50` and `rao`, as `experiments/erfi_limits_l*` did for Go1);
   history length (7 as in the paper, but state the resulting input size);
   policy size ([512, 512] as the paper's blind setup, or Playground's
   humanoid default if that is larger); `num_resets_per_eval` (Go1 trained
   with 5.6 s effective episodes, see `docs/eval_design.md` section 2.2;
   decide explicitly and record it); `tracking_sigma` (point 8 above).
e. States what is not possible or not meaningful on a humanoid and why: the
   paper has no humanoid results, so there is no reference curve; the
   quadruped protocol's "progress along initial heading" is fine, but the
   push direction should be reported split into sagittal and lateral because
   humanoids are far weaker laterally; the `phase` observation means the
   policy is not strictly the paper's blind state.
f. Lists the evaluation dimensions to reuse from `docs/eval_design.md` (own-
   terrain protocol, relief sweep, action delay, base torque, learning
   reliability) and the pod time for each.
g. Ends with open questions for me.

## Step 2: implementation

- Generalise `_ERFIMixin` so the joint slices and state assembly are derived
  from the base task's layout (a small per-robot descriptor: joint slice
  start, whether command precedes joints, whether a phase tail exists). Keep
  `Go1JoystickERFI` and `A1JoystickERFI` byte-for-byte equivalent in
  behaviour: add a test that their observations before and after the change
  match on a fixed key.
- Add the humanoid env class in `src/rl_locomotion/envs/<robot>/` following
  the A1 pattern, register it in `erfi.ROBOTS` / `ENV_CLASSES`, and extend
  `condition_config` validation.
- Accept a per-joint torque limit (array) in `cfg.erfi.rfi_lim` / `rao_lim`;
  scalar still allowed. Add a helper that measures stance torques and writes
  the limit vector into the experiment config.
- Extend `perturb.PROTOCOL` scaling by robot mass, read from the model, so
  the levels in the YAML are fractions where that is cleaner; do not change
  the numeric levels used by the Go1 configs.
- New configs `configs/experiment/erfi_study_<robot>.yaml` and
  `_rough.yaml`; add the robot to `runpod/run_all_studies.sh` via `STUDIES`.
- Tests in `tests/test_<robot>.py` that run on CPU in under two minutes:
  env builds for flat and rough, state size as documented, every condition
  steps with `qfrc_applied` cleared afterwards, torque limit array is applied
  per joint, protocol runs batched with a zero policy, Go1 observations
  unchanged.

## Step 3: pod commands and smoke (stop for review)

Print the exact commands: a `--smoke` pipeline check, the 20 M timing run, the
torque-limit sweep, the full study, and the evaluation, all through
`runpod/run_all_studies.sh` where possible. Do not launch anything.

## Constraints

- Ground every choice in a file, a measurement, or a paper section; no
  generic RL advice.
- Reuse the run-directory format and the config-driven env (`env_config.json`
  must fully determine the humanoid env, as it does for Go1).
- Ask me only when two readings of the task lead to materially different
  work.
