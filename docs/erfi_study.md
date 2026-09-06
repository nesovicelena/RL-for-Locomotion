# Go1 ERFI study: blind flat-ground robustness

Replication target: Campanaro, Gangapurwala, Merkt, Havoutis, *Learning and
Deploying Robust Locomotion Policies with Minimal Dynamics Randomization*
(arXiv:2209.12878). The paper's quantitative results are on ANYmal C with a
perceptive policy evaluated on stairs; its Unitree A1 setup (Sec. VI-B) is blind
and flat-ground but hardware-only. This study makes the blind setup quantitative
on the Go1, using the paper's evaluation protocol (Sec. VII) on flat ground.

## Method under test

Random torques are added on top of the PD actuator at the 12 joints, only
during training (`src/rl_locomotion/envs/erfi.py`):

| Scheme | Torque added |
|---|---|
| RFI | `tau_r ~ U(-lim, lim)` redrawn every control step |
| RAO | `tau_o ~ U(-lim, lim)` drawn once per episode |
| ERFI-C | both |
| ERFI-50 | each episode is RFI or RAO with probability 1/2 |

Limits default to 7 Nm. The paper used 20-40 Nm on ANYmal C against an 80 Nm
actuator; 7 Nm is the same ~30% fraction of Go1's 23.7 Nm hip/thigh limit.

## Training conditions

| Condition | ERFI | Playground dynamics randomization |
|---|---|---|
| `none` | off | off |
| `dr` | off | on (friction, link masses, CoM, armature, qpos0) |
| `rfi`, `rao`, `erfi_c`, `erfi_50` | on | off |

Everything else is identical: Playground's Go1 joystick task and reward,
Playground's PPO hyperparameters (200M steps, 8192 envs), same seeds.

Observation follows the paper's A1 setup: gravity vector, base linear and
angular velocity, 7-step history of joint position errors and joint velocities,
previous action, command. 192 dimensions. Policy MLP [512, 512]. The critic
uses Playground's privileged state by default (`symmetric_critic: false`); the
deployed policy is unaffected by that choice.

## Evaluation protocol

ERFI is off at evaluation. Fixed command 0.5 m/s forward for 8 s. Success =
the robot did not fall and advanced at least 2.5 m along its initial heading.
50 episodes per level, one parameter altered at a time
(`src/rl_locomotion/eval/perturb.py`):

| Parameter | Levels | Training value | Paper (ANYmal C) |
|---|---|---|---|
| payload on trunk (kg) | 0 .. 6 | 0 | base mass 22-65 kg, nominal 27 |
| push on trunk, 3 s (N) | 0 .. 40 | 0 | 0-150 N |
| floor friction | 0.2 .. 0.8 | 0.6 | 0.2-0.8, nominal 0.5 |
| gravity (m/s²) | -18 .. -2 | -9.81 | same |
| Kp multiplier | 0.33 .. 1.5 | 1.0 | hardware test at Kp/3 |

Mass and force ranges are scaled by the mass ratio Go1 / ANYmal C (12.7 kg vs
about 50 kg). Both training and evaluation run in MuJoCo, so the paper's
sim-to-sim gap (Isaac Gym to RaiSim) is not reproduced.

## Running it

```bash
# pod, inside tmux
python scripts/train.py --config configs/experiment/erfi_study.yaml
python scripts/eval.py  --config configs/experiment/erfi_study.yaml --plot
```

`notebooks/remote/erfi_study.ipynb` wraps the same commands with a smoke test,
progress table, training curves, and a rendering cell. Runs live in
`/workspace/experiments/erfi_study/<condition>/seed<k>/`; finished runs are
skipped on re-run, so the sweep resumes after a pod restart.

Outputs: `results.csv` (run x parameter x level), `summary_success_rate.csv`,
`success_rate_curves.png`, `fall_rate_curves.png`, `progress_m_curves.png`.

## Known limitations

- The evaluation uses the JAX MJX backend, since the perturbed model is passed
  as a traced argument. Training uses Warp. Both simulate the same model.
- The push direction is random in the horizontal plane; the paper does not
  state its direction.
- No Kinova-arm variant. The payload sweep is the closest analogue.

## Log

Add one entry per sweep: date, commit, GPU, anything changed from the config,
and the summary table.
