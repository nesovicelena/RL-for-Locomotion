# ERFI study on a Playground humanoid: design

Status: design (Sections 1-7) reviewed and implemented (Sections 8-9). The
primary study is `configs/experiment/erfi_study_bh_rough.yaml` (rough terrain,
as decided at review); the flat study is `erfi_study_bh.yaml`. Nothing under
`experiments/` has been changed. Every number below was read from the
installed Playground 0.14.2 (`mujoco_playground/_src/locomotion/
berkeley_humanoid/`), from this repository, or measured on the laptop CPU with
the MuJoCo C engine (a 2 s stand; no training, no evaluation). Measurements are
marked "measured"; the script that produced them is described in 4.1 so the
pod can repeat them.

Decisions taken at review (open questions of Section 7): torque-limit route as
in 4.1 (Q1); training pushes off (Q2); payload up to 20 % as fractions in the
YAML (Q3); 3 seeds (Q4); `num_resets_per_eval` 10 (Q5); `tracking_sigma` 0.5
(Q6); Playground's Berkeley PPO entry (Q7); `history_target_error` on (Q8);
push direction as two panels (Q9); G1 not prepared (Q10).

---

## 1. Robot: Berkeley Humanoid first, G1 as the stretch target

Berkeley Humanoid (`BerkeleyHumanoidJoystickFlatTerrain` / `RoughTerrain`) is
the closest humanoid to what the Go1 code already handles: 12 actuators and
18 velocity DOFs, exactly Go1's `nu`/`nv`, so the `qfrc_applied` injection at
`nv - nu = 6`, the 12-wide RAO offset, the randomiser's `shape=(12,)` draws and
the `[512, 512]` policy carry over without a size change anywhere. It is the
lightest of the three (16.06 kg total, torso 5.38 kg, measured from
`body_subtreemass`), its scenes mirror Go1's exactly (plane with friction 0.6 /
the same 20 x 20 m, 5 cm heightfield asset with friction 1.0), its domain
randomiser uses Go1's index conventions (floor geom 0, torso body 1, DOFs 6:),
and Playground trains it in 150 M steps with the same Brax PPO family, so the
200 M budget of the Go1 study is known to be enough. G1 (33.3 kg, 29 actuators)
is the paper-like scale and has the extra self-contact termination that makes
"fall" stricter, but its randomiser uses `pair_friction` and torso body 16
(`torso_link`), its state is 103-dim, and each run will cost several times
more; it is the second study once the Berkeley one works, not the first.

## 2. What the mixin must generalise (from `joystick.py` of the chosen task)

All facts below are for Berkeley Humanoid unless stated; Go1 in brackets.

### 2.1 Model and task constants

| quantity | Berkeley Humanoid | source |
|---|---|---|
| `nu` / `nv` / `nq` / `nbody` | 12 / 18 / 19 / 14 [12 / 18 / 19 / 14] | measured |
| actuator (= joint) order | LL_HR, LL_HAA, LL_HFE, LL_KFE, LL_FFE, LL_FAA, then LR_* | measured |
| total mass / torso mass | 16.06 kg / 5.38 kg [12.74 / 5.20] | measured |
| Kp per actuator | 35 on ten joints, **1 on the two ankle-roll joints (FAA)** [35 all] | `berkeley_humanoid_mjx_feetonly.xml` classes `humanoid`, `faa` |
| Kd | `dof_damping` 1.5 (FAA 0.1), `biasprm[:, 2] = 0` [Kd 0.5 via biasprm] | measured |
| `actuator_forcerange` | 0 (unlimited) on all actuators | measured |
| **`jnt_actfrcrange`** (joint-level clamp of the actuator force, `jnt_actfrclimited` True) | +-20 (HR, HAA), +-30 (HFE, KFE), +-20 (FFE), +-5 (FAA) Nm [Go1 clamps via `actuator_forcerange` 23.7 / 35.55] | measured; XML classes `hxx`, `hfe`, `kfe`, `ffe`, `faa` |
| `ctrl_dt` / `sim_dt` / substeps | 0.02 / 0.002 / **10** [0.02 / 0.004 / 5] | `default_config`, `env.n_substeps` |
| `action_scale` | 0.5 [0.5] | `default_config` |
| floor friction flat / rough | 0.6 / 1.0 [0.6 / 1.0] | measured, `scene_mjx_feetonly_*.xml` |
| keyframe `home` height flat / rough | 0.515 m / 0.56 m [0.278 / 0.35] | measured |
| heightfield | `hfield_size = (10, 10, 0.05, 1.0)`, same as Go1 | measured |
| collision | **feet only** (two boxes `l_foot1`, `r_foot1`, 4 corners each = 8 contacts standing); torso and legs have `contype 0` | XML class `collision`; measured `ncon = 8` |
| `episode_length` | 1000 steps = 20 s [1000] | `default_config` |
| command ranges | `lin_vel_x`, `lin_vel_y`, `ang_vel_yaw` all U(-1, 1), all zero with p = 0.1 [Go1 amplitudes 1.5 / 0.8 / 1.2] | `default_config`, `sample_command` |
| `tracking_sigma` | **0.5** [0.25 v1, 0.1 v2/v3] | `reward_config` |
| built-in disturbance | `push_config` (velocity kicks 0.1-2.0 m/s every 5-10 s, **enabled by default**) [`pert_config`, disabled by default] | `default_config` |
| PPO defaults (`locomotion_params.brax_ppo_config`) | 150 M steps, 15 evals, `entropy_cost` 0.005, `clipping_epsilon` 0.2, `num_resets_per_eval` 10 (inherited), nets (512, 256, 128) / (512, 256, 128) [200 M, 10 evals, entropy 0.01, Brax default clipping] | `config/locomotion_params.py` |
| domain randomiser | floor geom 0 friction U(0.4, 1.0); `dof_frictionloss[6:]` x U(0.9, 1.1); `dof_armature[6:]` x U(1.0, 1.05); all masses x U(0.9, 1.1); **torso mass + U(-1, 1) kg; `qpos0[7:]` + U(-0.05, 0.05)**; no CoM jitter [Go1: same first four, plus torso `body_ipos` jitter, no torso-mass or qpos0 term] | `berkeley_humanoid/randomize.py` |
| observation sizes (Playground) | state 52, privileged 114 [48 / 123] | measured |

### 2.2 State layout and the ERFI state size

Playground's Berkeley state (`_get_obs`):

```
0:3   linvel (noisy)      3:6  gyro (noisy)     6:9  gravity (noisy)
9:12  command
12:24 joint_angles - default_pose (noisy, 12)
24:36 joint_vel (noisy, 12)
36:48 last_act (12)
48:52 phase = [cos(phase_L), cos(phase_R), sin(phase_L), sin(phase_R)]
```

Go1: `linvel | gyro | gravity | joints | joint_vel | last_act | command`, so
`_JOINT_POS = slice(9, 21)` and `_JOINT_VEL = slice(21, 33)` are wrong here.
The mixin needs a per-robot descriptor with: `joint_pos_start` (9 for Go1/A1,
12 for the humanoids), joint slices derived as `slice(s, s + nu)` and
`slice(s + nu, s + 2 nu)`, and `phase_tail` (0 for Go1/A1, 4 for Berkeley;
G1 and T1 also end with a 4-entry phase). The reordered ERFI state becomes

```
gravity 3 | linvel 3 | gyro 3 | joint-pos history 12 H | joint-vel history 12 H
| last_act 12 | command 3 | phase 4
```

which is **196** dimensions at H = 7 (Go1: 192), 52 at H = 1. The privileged
state stays Playground's 114 (it already embeds the 52-dim state, so the critic
sees the phase), 127 with `critic_sees_offset` (114 + 12 + 1).

### 2.3 Everything else the code has to handle

1. **`_get_obs` signature.** Berkeley's is `_get_obs(data, info, contact)`;
   Go1's is `_get_obs(data, info)`. The mixin override must take `*args` and
   pass them through. The mixin's own call in `reset()` (the
   `critic_sees_offset` re-observation) has no `contact`; it must recompute it
   from the `*_floor_found` sensors as the base `reset` does, or the base
   observation must be recomputed through a helper that owns that logic.
2. **Info keys.** Berkeley adds `phase`, `phase_dt`, `push`, `push_step`,
   `push_interval_steps`, `feet_air_time`, `last_contact`, `swing_peak`, and
   lacks nothing the mixin reads (`rng`, `command`, `last_act`,
   `motor_targets`). The mixin adds `applied_act`, `erfi_offset`,
   `erfi_use_rfi`, `joint_pos_hist`, `joint_vel_hist` as before.
3. **Built-in pushes.** `perturb.eval_env_config` sets
   `cfg.pert_config.enable = False`; Berkeley has `push_config` instead, and it
   is *on* by default. Two changes: the eval config must disable whichever of
   the two the task has (otherwise the protocol's episodes get random 2 m/s
   velocity kicks), and the study must decide the training value (4.5).
4. **Config root.** `erfi.default_config()` starts from Go1's
   `joystick.default_config()`. It must start from the chosen robot's task
   default (Berkeley's carries `push_config`, `gait_freq` sampling in `reset`,
   `soft_joint_pos_limit_factor`, its own reward scales) and add the ERFI keys
   on top. Berkeley's config already has a `history_len = 1` key that its code
   never reads; our `history_len = 7` overwrites it, no conflict.
5. **Torque limit as a vector.** `rfi_lim` / `rao_lim` become a 12-vector in
   the config (JSON list in `env_config.json`); `jax.random.uniform(minval=-lim,
   maxval=lim)` broadcasts, so `_sample_erfi` and `step` need only
   `jp.asarray`. Scalars remain valid for Go1/A1.
6. **Injected torque bypasses the joint clamp.** MJX clamps `qfrc_actuator`
   to `jnt_actfrcrange` after the actuator moment (`mjx/_src/forward.py`,
   `# clamp qfrc_actuator`); `qfrc_applied` is added later and is never
   clamped. This is the same situation as on Go1, where `actuator_forcerange`
   clamps only the actuator. The report must say that the perturbation is
   not limited by the motor model on either robot.
7. **Domain randomiser selection.** `erfi.domain_randomizer` returns Go1's
   function by name for every robot. For Berkeley it must return
   `registry.get_domain_randomizer("BerkeleyHumanoidJoystick*Terrain")`. The
   index conventions match Go1's, so the machinery is reusable, but the
   *content* differs (2.1): the `dr` condition on Berkeley is "Playground's
   Berkeley randomiser", not Go1's, and the report table must list its terms.
   For the record: T1 also uses torso body 1 but friction U(0.2, 0.6) and mass
   x U(0.98, 1.02); G1 uses `pair_friction[0:2, 0:2]` and torso body 16.
8. **Termination and the fallen pose.** `done` = gravity z < 0 or NaN. With
   feet-only collision the fallen torso passes through the plane (measured: a
   2 s zero-action stand ended at torso z = -0.52 m, upside down). The
   protocol freezes progress at `done`, so results are unaffected; videos and
   any "continue after done" code path are not meaningful. A robot kneeling
   with its torso upright is not `done` and its legs sink through the floor,
   so the protocol should also report `low_base` (torso z below half the
   spawn height, 0.26 m) as a second failure flag and count it as a fall.
9. **Substeps.** RFI is drawn once per control step and held across the ten
   substeps (Go1: five). The paper resamples "at each impedance control
   update step"; the deviation is the same as in the Go1 report, only with a
   coarser ratio.
10. **Warp on the pod, JAX locally.** Berkeley's default `impl` is `warp`;
    `mujoco_warp` is not installed on the laptop, so every CPU test uses
    `impl="jax"` as `tests/test_a1.py` does.
11. **Rough-terrain contact budget.** `condition_config` raises `njmax` to
    128 on rough terrain; Berkeley's defaults are `naconmax 8*8192`,
    `njmax 60`. Berkeley's `Joystick.__init__` does not touch them (Go1's
    does), so the mixin's "restore ours" logic is harmless here.

### 2.4 Protocol scaling by mass (`perturb.PROTOCOL`)

The Go1 levels are fixed numbers; they must not change. For the humanoid the
levels are stated as fractions and resolved against the model at eval time
(`body_subtreemass[0]`):

| parameter | scaling rule | Berkeley levels | Go1 equivalent (unchanged) |
|---|---|---|---|
| `payload_kg` | fraction of total mass, on `_torso_body_id` | {0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20} x 16.06 = {0, 0.40, 0.80, 1.20, 1.61, 2.41, 3.21} kg | 0-6 kg = up to 47 % |
| `push_N` | fraction of body weight m g, 3 s, from 1 s | {0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.32} x 157.5 N = {0, 6.3, 12.6, 18.9, 25.2, 31.5, 37.8, 50.4} N | 0-40 N = up to 0.32 of weight; paper 150 N on 50 kg = 0.31 |
| `torque_Nm` (D7) | fraction of m g x 1 m | {0, 0.016, 0.032, 0.064, 0.095, 0.127, 0.16} -> {0, 2.5, 5, 10, 15, 20, 25} N m | paper 75 N m on 50 kg; Go1 19 N m proposed |
| `friction` | absolute, unchanged | 0.2-0.8 flat (trained 0.6), 0.3-1.3 rough (trained 1.0) | same |
| `gravity` | absolute, unchanged | -2 to -18 | same |
| `kp_scale` | multiplier, unchanged; multiplies the per-actuator `gainprm[:, 0]` and `biasprm[:, 1]`, so the FAA joints go 1 -> 0.33..1.5 along with the rest | 0.33-1.5 | same |

The 20 % payload ceiling follows the prompt; Go1 used 47 % and saturated at
1.0 on flat ground (`docs/eval_design.md`, goal paragraph). A humanoid carries
the payload high on the torso, so 20 % is a much larger disturbance to balance
than 47 % was on a quadruped, but the ceiling is a guess until the first
`none` policy is evaluated; see open question 3.

**Push direction.** The Go1 protocol draws one random horizontal direction per
episode. For the humanoid the direction is a factor: add `push_axis` to
`EvalSpec` with values `random` (default, Go1 behaviour unchanged), `sagittal`
(+-x of the initial heading, sign random) and `lateral` (+-y). The humanoid
runs both named axes, 50 episodes each, and the report shows two push panels.

---

## 3. The study

**Conditions.** All six (`none`, `dr`, `rfi`, `rao`, `erfi_c`, `erfi_50`).
The randomiser check in 2.3 item 7 passes for Berkeley with the change of
which function is returned.

**Seeds.** 3 (0, 1, 2) to match Go1 and its figure code (mean line, min-max
band, walking-seed counts n/3). 5 seeds only if the 20 M timing run puts a
200 M run under about 10 min: 6 x 5 x 2 terrains = 60 runs would then be
under 10 h, which fits one overnight pod session like `run_all_studies.sh`
was written for. Above that, 3 seeds, and the fourth and fifth seed are a
follow-up on flat terrain only.

**Budget.** 200 M steps per run, 10 evals, as Go1 (`TrainSpec` defaults).
Playground's own Berkeley config stops at 150 M, so 200 M is known to
suffice for the nominal task. Timing is measured, not assumed: one
`none seed 0` run at 20 M steps with 2 evals; the wall time minus the first
eval's JIT time, times 10, is the 200 M estimate (`summary.json` `wall_s`,
`curve.json` `wall_s` per eval). Expectation before measuring: Go1 did 200 M
in 5.5 min (about 600 k steps/s); Berkeley has twice the substeps and eight
box-plane contact points instead of four sphere contacts, so 2.5-4x slower,
14-22 min per run, 4-7 h for 18 flat runs, similar again for rough. If the
measured time is above 30 min per run, drop to 150 M (Playground's budget)
and say so.

**Order.** Flat first, rough second, each through
`runpod/run_all_studies.sh` with `STUDIES="erfi_study_bh"` and
`"erfi_study_bh_rough"`. Within flat the order is dictated by the torque
limit (4.1): `none` x 3 seeds first (needed anyway), stance measurement,
50 M limit sweep, then the other five conditions.

**PPO settings.** `ppo.ppo_config` builds from
`brax_ppo_config("Go1JoystickFlatTerrain")` for every robot. For Berkeley it
should build from `brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")`
(Playground's tuned values: entropy 0.005, clipping 0.2) with the study's
overrides (200 M, 10 evals, the network sizes in 4.3). This is a change of
two PPO constants relative to the Go1 study and is recorded in
`ppo_config.json` as always; the report lists it in the "what differs from
Go1" table.

---

## 4. Parameters to tune and the measurement that decides each

### 4.1 Torque limit (per joint)

**What was measured (laptop, MuJoCo C engine, flat scene).**

- *Zero-action stand, Kp 35 as shipped, ctrl = home pose, 2 s:* the robot
  falls within 1 s (gravity z = -0.9998 at 2 s, torso below the plane). A
  plain PD on the home pose does not stand: the ankle-roll actuators have
  Kp 1 and no policy is holding balance. The prompt's "zero-action stand" is
  therefore **not usable** on this robot, and Playground ships no pretrained
  Berkeley checkpoint in the pip package (no `.onnx`/params files under
  `mujoco_playground/`).
- *Static stance with a stiff hold* (same pose, Kp 200 / Kd 10 and Kp 500 /
  Kd 20, 2 s, RMS of `actuator_force` over 0.5-2 s; both gain settings agree
  to within 0.5 Nm, so this is the gravity-compensation torque of the home
  pose, not a gain artefact):

  | joint | HR | HAA | HFE | KFE | FFE | FAA |
  |---|---|---|---|---|---|---|
  | static stance RMS (Nm) | 0.1-0.4 | 0.5-0.7 | 0.5-0.7 | **6.7-7.2** | 0.4-1.0 | 0.4-0.5 |
  | 0.5 x static | ~0.2 | ~0.3 | ~0.3 | ~3.5 | ~0.4 | ~0.2 |
  | 0.1 x `jnt_actfrcrange` (Go1's rule) | 2.0 | 2.0 | 3.0 | 3.0 | 2.0 | 0.5 |

  Standing, the torque is almost entirely in the knees. Half the static
  stance vector would make RFI/RAO a knee-only perturbation, unlike the
  paper's uniform 20 Nm on every ANYmal joint (VIII, Fig. 5i) and unlike Go1's
  uniform 2.5 Nm. Walking torques at the hips and ankles are far larger than
  standing ones, so the static measurement underestimates the relevant
  "stance torque" for a walking humanoid.
- *Why not Go1's 10 %-of-limit rule alone:* the FAA actuator's whole PD
  authority is Kp 1 x 0.52 rad (full joint range) = 0.5 Nm. A 0.5 Nm
  perturbation would exceed what the controller can counter on that joint;
  the joint's `jnt_actfrcrange` of 5 Nm says nothing about that.

**Decision procedure (measurement, not a guess).**

1. Provisional vector for the smoke and timing runs only:
   0.1 x `jnt_actfrcrange` = [2, 2, 3, 3, 2, 0.5] x 2 legs. Recorded as
   provisional in the config comment; never used for a reported result.
2. Train `none` seeds 0-2 at 200 M on flat (needed for the study anyway).
3. Helper `scripts/measure_stance_torque.py`: load the `none` policy that
   walks best at nominal (protocol row payload 0), roll 50 episodes of 8 s at
   0.5 m/s on the JAX backend with ERFI off (the protocol's own rollout),
   record `data.actuator_force` every control step while not `done`, and
   write the per-joint RMS and 0.5 x RMS into
   `configs/experiment/erfi_study_bh.yaml` (`rfi_lim`, `rao_lim` as lists)
   with the measurement provenance (run dir, checkpoint, date) as a comment.
   The privileged state already carries `actuator_force`, so this is a
   minute of pod time. The fraction 0.5 is the paper's "half a stance
   torque" as the Go1 study interpreted it.
4. Sweep, as `experiments/erfi_limits_l*` did: fractions 0.25 / 0.5 / 1.0 of
   the measured vector, conditions `erfi_50` and `rao`, seed 0, 50 M steps
   (6 runs). Decision rule, read from `curve.json` and the nominal protocol
   row: choose the largest fraction at which both `erfi_50` and `rao` leave
   the standing plateau within 50 M (reward exceeds the plateau by the
   D4 margin) and `erfi_50` walks at nominal (progress >= 2.5 m). Paper
   Fig. 5i says larger limits buy robustness as long as training still
   succeeds; Go1's sweep chose 2.5 Nm because 7 Nm left RAO/ERFI standing.
5. Symmetry: the left and right legs are averaged so the vector is
   left-right symmetric (the model is; the policy's gait need not be).

### 4.2 History length

7, as the paper (VI-B: 84 = 7 x 12 joint-position errors and 84 joint
velocities). Resulting policy input: **196** (2.2). `history_target_error`
follows the recipe: the first Berkeley study uses the v3 recipe's
`q* - q` (the paper's quantity), since there is no v1/v2 legacy to stay
compatible with on this robot.

### 4.3 Network size

Policy [512, 512] as the paper's blind setup and the Go1 study; critic
[512, 256, 128], asymmetric on the 114/127-dim privileged state. Playground's
Berkeley default is [512, 256, 128] for the policy, which has fewer
parameters than [512, 512] at a 196-dim input (512 x 196 + 512 x 256 + 256 x 128
versus 512 x 196 + 512 x 512), so the paper's size is not smaller than the
tuned default; keep [512, 512] for comparability with Go1.

### 4.4 `num_resets_per_eval` and the effective episode length

Brax runs `num_evals_after_init x num_resets_per_eval` epochs and fully
resets every env between epochs (`docs/eval_design.md` 2.2). With 8192 envs,
unroll 20, 200 M steps, 10 evals and `num_resets_per_eval = 10` (Berkeley
inherits 10 from the base config; G1/T1 override to 1) the effective episode
is 280 control steps = **5.6 s**, exactly Go1's. Playground's own Berkeley
setting (150 M, 15 evals, 10 resets) trains on 2.8 s episodes and still
produces a walking policy. Decision: keep 10, so the Berkeley and Go1
policies trained on the same 5.6 s horizon, and record it in `ppo_config.json`
and the report. The protocol (8 s) is longer than any training episode, as on
Go1. Open question 5 asks whether you prefer G1/T1's 1 (20 s episodes).

### 4.5 `tracking_sigma` and the standing return

Per-step tracking reward for a robot standing still under a 0.5 m/s command,
`exp(-0.25 / sigma)`, scale 1.0:

| sigma | reward | used by |
|---|---|---|
| 0.5 | 0.607 | Berkeley default |
| 0.25 | 0.368 | Go1 v1 |
| 0.1 | 0.082 | Go1 v2 / v3 |

Berkeley's reward differs from Go1's in two ways that already act against the
standing optimum: `stand_still` has scale 0 (Go1: -1.0, i.e. Go1 *penalised*
moving at zero command, not standing at nonzero command), and `feet_phase`
(scale 1.0, `exp(-err / 0.01)` on foot height against the gait clock) is not
gated on the command (the code carries a TODO for that), so a standing policy
forfeits up to 1.0 per step no matter what the command is. Decision: keep
Playground's 0.5 for the first flat study and measure the standing plateau
from the `none` runs' first three evals (D4 definition). Lowering to 0.1 is a
second recipe if RAO/ERFI seeds stall, not a default; the v2 lesson came from
a reward without a phase term.

### 4.6 Built-in velocity pushes during training

Playground trains Berkeley with `push_config.enable = True` (2.1). That is a
disturbance-training method in its own right and would blur the `none`
condition. Decision: `push_config.enable = False` for all six conditions, so
`none` means no randomisation and no disturbance, as in the paper. Consequence
to state: our `none` is harder to train than Playground's Berkeley baseline;
if `none` fails to walk on 3 seeds, that is itself a result, and a seventh
condition `push` (Playground's default) can be added without touching the
other six.

---

## 5. What is not possible or not meaningful on a humanoid

1. **No reference curve.** The paper has no humanoid results (VI: ANYmal C
   and A1 only), so the study can only test whether the *ordering* of
   conditions from the Go1 study and Fig. 5 recurs; no number can be
   compared with the paper.
2. **The state is not the paper's blind state.** The four `phase` entries
   are a gait clock the task's reward depends on (`feet_phase`, `feet_air_time`
   thresholds). Removing them would change the task, so they stay; the report
   calls the input "the paper's 192-dim blind state plus the task's 4-dim
   gait clock", 196 in total.
3. **Push direction is a factor.** The Go1 protocol's random direction is
   fine for a quadruped; a biped is far weaker laterally, so sagittal and
   lateral pushes are separate sweeps (2.4). "Progress along initial heading"
   stays as the success criterion.
4. **Fall criterion.** `done` fires only when the torso passes horizontal.
   Together with feet-only collision this misses "kneeling / legs through the
   floor" states (2.3 item 8), hence the extra `low_base` flag. G1 would add
   feet-shin and foot-foot self-contact terminations; on Berkeley no such
   sensors exist, so foot-foot collision is not even simulated (feet collide
   only with the floor).
5. **Torque-limit fraction is not transferable.** The paper's 20 Nm and Go1's
   2.5 Nm were single scalars; here the limit is a vector shaped by a measured
   torque profile, so "same fraction" is the only cross-robot statement that
   can be made.
6. **Kp/3 hardware test (Fig. 4h)** stays a simulation sweep; the FAA joints
   at Kp 1 x 0.33 are effectively passive at the low end, so the Kp panel
   below 0.5 measures ankle-roll compliance as much as gain robustness.
7. **Cross-simulator evaluation** remains Warp-train / JAX-eval, as in
   `docs/eval_design.md` 1.2 and D8.

---

## 6. Evaluation dimensions reused from `docs/eval_design.md`

Pod time scaled from the Go1 figure of about 1 min per run for the 34-level
protocol including compile; Berkeley's 10 substeps and bigger contact set make
the physics 2-3x slower, and the compile is per policy. Per-run protocol time
assumed 2-3 min. 18 runs per terrain.

| dim | what | change needed for the humanoid | pod time (flat + rough) |
|---|---|---|---|
| D1 | own-terrain protocol: payload, push (sagittal, lateral), friction, gravity, Kp, plus `low_base` | mass-scaled levels, `push_axis`, `low_base` flag | 36 runs x ~3 min = ~2 h |
| D2 | relief sweep 0-12.5 cm on rough-trained policies | none beyond D2's own implementation | 18 x 6 levels, ~30 min |
| D3 | flat-trained on rough at friction 0.6 and 1.0 | none | 18 x 6 x 2, ~30 min |
| D4 | learning reliability from `curve.json` (walking seeds >= 1 m and >= 2.5 m, steps to walk against the `none` plateau) | plateau measured per robot | 0 |
| D5 | nominal gait quality (progress, `tracking_rmse_alive`, fall, `low_base`) | column already planned | 0 |
| D6 | action delay 0-4 steps (0-80 ms) | none | 36 x 5, ~20 min |
| D7 | base yaw torque 0-25 N m, 1 s | mass-scaled levels | 36 x 7, ~20 min |
| D8 | classic-MuJoCo nominal check | optional, decide after D1-D7 | CPU |

Total evaluation about 4 h of pod time for both terrains, dominated by D1.

---

## 7. Open questions

1. **Torque-limit route.** Agree with 4.1: provisional 0.1 x `jnt_actfrcrange`
   for smoke/timing, then 0.5 x the walking RMS of the best `none` policy,
   then the 0.25 / 0.5 / 1.0 sweep? The alternative that needs no `none`
   policy first is the static stance vector, which is knee-only.
2. **Training pushes off** (4.6) in all six conditions, or keep Playground's
   default on so the baseline matches Playground's published Berkeley result?
3. **Payload ceiling 20 %** versus Go1's 47 %; and whether the humanoid levels
   should be stated in the YAML as fractions (proposed) or in kg/N.
4. **Seeds.** 3 now, 5 only if the timing run says under 10 min per 200 M run?
5. **`num_resets_per_eval`.** 10 (5.6 s episodes, same as Go1) as proposed, or
   1 (20 s, Playground's G1/T1 choice)?
6. **`tracking_sigma` 0.5** (Playground default) for the first study, with 0.1
   only as a fallback recipe?
7. **PPO base config** from Playground's Berkeley entry (entropy 0.005) rather
   than Go1's (0.01)?
8. **`history_target_error = True`** (v3 recipe, the paper's quantity) from the
   start on this robot, since there is no v1/v2 legacy here?
9. **Reporting split of push direction** as two panels (proposed) or one panel
   with two line styles per condition?
10. Whether G1 should be prepared in the same code change (descriptor, mass
    scaling, `pair_friction` handling in the protocol's friction sweep) or left
    entirely for later.

---

## 8. Implementation notes (Step 2)

What changed, and what did not.

- `src/rl_locomotion/envs/erfi.py`: `RobotLayout` (joint offset, phase tail,
  built-in-disturbance key, Playground env names, task default config) per
  robot in `LAYOUTS`; joint slices derived from `nu`; `_get_obs(data, info,
  *args)` passes Berkeley's `contact` through; `_obs_extra_args` hook for the
  re-observation in `reset`; `default_config(robot)`; per-joint limit vectors
  (`set_torque_limits`, validated against `nu` at build time);
  `playground_env_name`; `domain_randomizer` returns the robot's own function;
  `total_mass` and `spawn_height` properties for the protocol. The RFI/RAO
  draws keep the exact `uniform(minval=-lim, maxval=lim)` expression, so Go1
  and A1 are bit-for-bit as before: `tests/test_bh.py` compares 11 steps of
  both against `tests/golden/erfi_go1_a1_obs.npz`, recorded with the
  pre-change mixin.
- `src/rl_locomotion/envs/bh/`: names Playground's Berkeley `Joystick` as the
  base class and records the joint order, the joint clamps and the
  provisional limit vector.
- `src/rl_locomotion/eval/perturb.py`: `push_N_sagittal` / `push_N_lateral`
  (traced axis code, one compile), `EvalSpec.level_fractions` resolved against
  the model's mass, `EvalSpec.low_base_fraction`, new columns
  `low_base_rate`, `tracking_rmse_alive`, `robot`, `robot_mass_kg`;
  `eval_env_config` disables `push_config` as well as `pert_config` and ERFI.
  Go1 levels, default params and default success criterion are unchanged
  (pinned by a test).
- `src/rl_locomotion/training/ppo.py`: list-valued `rfi_lim` / `rao_lim` in
  `TrainSpec`; the PPO base config comes from the robot's Playground entry.
- `scripts/train.py` (`--rfi-lim` / `--rao-lim` take one or `nu` numbers),
  `scripts/eval.py` (accepts the push-axis params),
  `scripts/measure_stance_torque.py` (new; 4.1 step 3), the two configs,
  `runpod/run_all_studies.sh` (`STUDIES` default now includes both Berkeley
  studies, rough first).
- After code review (8.1): three changes to the humanoid pipeline, none to
  Go1/A1.
  1. **Termination.** `BerkeleyHumanoidJoystickERFI._get_termination` adds
     `base z < cfg.min_base_height` (default 0.2575 m, half the flat spawn
     height; 0 disables) to the task's gravity/NaN rule. Reason: the scene
     collides feet only, so a buckled robot sinks through the floor with its
     torso upright and would otherwise keep training. A true terminal for
     Brax; recorded in `env_config.json`.
  2. **Foot slip.** Playground's `_cost_feet_slip` multiplies the *base*
     velocity by the contact flags (`joystick.py:603-609`), a penalty on
     walking speed during stance. The ERFI class computes it from the foot
     linear-velocity sensors instead (the ones `_cost_feet_clearance` reads).
     Scale unchanged (-0.25). Both 1 and 2 differ from Playground's published
     Berkeley task and go in the report's "what differs" table.
  3. **Protocol start.** `EvalSpec.nominal_reset` (on in both humanoid
     configs, off for Go1) replaces the reset's joint scaling U(0.5, 1.5) and
     base kick by the keyframe pose at rest, keeping the random base position
     and yaw; the ERFI history is refilled from the new reading. Paper VII:
     "always deployed at the same position".
- Not implemented, by scope: D7 base torque, D8 classic-MuJoCo check, the D2
  relief sweep as a traced level (evaluation extras from `docs/eval_design.md`
  that the Go1 studies also still lack); G1.

## 9. Pod commands (Step 3)

Everything below runs on the pod inside `tmux`, from the repository root, with
the same output root the overnight driver uses so finished runs are skipped on
restart. Nothing here has been launched.

```bash
export RL_EXPERIMENTS_DIR=/workspace/experiments/redo
CFG=configs/experiment/erfi_study_bh_rough.yaml
```

**9.1 Smoke (pipeline check, provisional limits).** Two conditions, one seed,
2 M steps each, then the protocol at 2 episodes per level on the smoke runs.
Expect a few minutes per run.

```bash
python scripts/train.py --config $CFG --conditions none erfi_50 --seeds 0 --smoke
python scripts/eval.py  --config $CFG --runs $RL_EXPERIMENTS_DIR/erfi_study_bh_rough_smoke --n-episodes 2 --plot
# or every condition and seed of the study at 2 M steps, through the driver:
SMOKE=1 STUDIES="erfi_study_bh_rough" bash runpod/run_all_studies.sh
```

Check: `results.csv` has rows for `push_N_sagittal` and `push_N_lateral`, a
`low_base_rate` column, payload levels 0-3.21 kg, push levels 0-50.4 N;
`env_config.json` shows `robot: bh`, `task: rough_terrain`, twelve-entry
`rfi_lim`, `push_config.enable: false`; `ppo_config.json` shows
`entropy_cost 0.005`.

**9.2 Timing run (decides seeds and budget, Section 3).**

```bash
python scripts/train.py --config $CFG --conditions none --seeds 0 \
    --num-timesteps 20_000_000 --num-evals 2 --out erfi_study_bh_rough_timing
R=$RL_EXPERIMENTS_DIR/erfi_study_bh_rough_timing/none/seed0
python -c "import json; c=json.load(open('$R/curve.json')); s=json.load(open('$R/summary.json')); \
jit=c[0]['wall_s']; t=s['wall_s']; print(f'20 M: {t/60:.1f} min, JIT {jit/60:.1f} min -> 200 M about {(t-jit)*10/60:.0f} min per run')"
```

Under 10 min per 200 M: 5 seeds (edit `seeds:` in both configs). Over 30 min:
`NUM_TIMESTEPS=150000000` on the driver, and say so in the report.

**9.3 `none` seeds, then the stance-torque measurement (4.1 steps 2-3).**

```bash
python scripts/train.py --config $CFG --conditions none          # seeds 0-2, 200 M each
python scripts/eval.py  --config $CFG --params payload_kg        # nominal row = payload 0
# pick the seed with the largest progress_m at payload 0 in
#   $RL_EXPERIMENTS_DIR/erfi_study_bh_rough/results.csv, then:
python scripts/measure_stance_torque.py $RL_EXPERIMENTS_DIR/erfi_study_bh_rough/none/seed<k> \
    --fraction 0.5 --write configs/experiment/erfi_study_bh_rough.yaml configs/experiment/erfi_study_bh.yaml
git diff configs/experiment/    # rfi_lim / rao_lim now carry the measured vector and its provenance
rm $RL_EXPERIMENTS_DIR/erfi_study_bh_rough/results.csv   # so 9.5 evaluates all 18 runs in one pass
```

The `none` runs are unaffected by the limit (ERFI is off in that condition), so
the full study reuses them; their `env_config.json` records the provisional
vector, which the report notes.

**9.4 Torque-limit sweep (4.1 step 4): 0.25 / 0.5 / 1.0 x the measured vector,
`erfi_50` and `rao`, seed 0, 50 M steps.**

```bash
for f in 0.25 0.5 1.0; do
  LIM=$(python -c "import yaml,sys; f=float(sys.argv[1]); v=yaml.safe_load(open('$CFG'))['train']['rfi_lim']; print(' '.join(f'{f*x:.3f}' for x in v))" $f)
  python scripts/train.py --config $CFG --conditions erfi_50 rao --seeds 0 \
      --num-timesteps 50_000_000 --num-evals 5 --rfi-lim $LIM --rao-lim $LIM --out erfi_limits_bh_f$f
  python scripts/eval.py --config $CFG --runs $RL_EXPERIMENTS_DIR/erfi_limits_bh_f$f --params payload_kg gravity
done
```

Decision rule (4.1): the largest fraction at which both conditions leave the
`none` standing plateau within 50 M (`curve.json`) and `erfi_50` reaches
2.5 m at nominal. If it is not 0.5, rerun 9.3's `--write` with that
`--fraction`.

**9.5 Full study, rough terrain (primary), then flat.**

```bash
STUDIES="erfi_study_bh_rough" bash runpod/run_all_studies.sh     # trains the 15 remaining runs, then evaluates all 18
STUDIES="erfi_study_bh"       bash runpod/run_all_studies.sh     # flat, all 18
# both, unattended, pod stopped afterwards:
STUDIES="erfi_study_bh_rough erfi_study_bh" STOP_POD=1 bash runpod/run_all_studies.sh
```

**9.6 Evaluation on its own** (the driver already runs it; ERFI, RAO and the
task's pushes are off in every evaluation env, `perturb.eval_env_config`).

```bash
python scripts/eval.py --config $CFG --plot                            # own terrain, both push axes
python scripts/eval.py --config $CFG --task flat_terrain --plot        # rough-trained policies on the flat scene (D3)
python scripts/eval.py --config $CFG --terrain-amplitude 0.10 --plot   # 10 cm relief (D2, one level per call)
```

**9.7 Bring the tree back.**

```bash
rsync -avz --progress --exclude 'params_0*' -e 'ssh -p <port>' root@<ip>:/workspace/experiments/redo/ experiments/redo/
```
