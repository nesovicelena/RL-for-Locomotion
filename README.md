# RL-for-Locomotion

Reinforcement learning for legged locomotion, built on
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) and
Brax/MJX.

Development happens locally; training happens on a cloud GPU instances.

## Layout

```
configs/       experiment definitions (env / algo / experiment)
src/           all real logic, installed as the `rl_locomotion` package
notebooks/     local/ = no GPU, remote/ = runs on the cloud pod
scripts/       headless entrypoints and tools
runpod/        bootstrap, sync, result retrieval
experiments/   run outputs and figures (gitignored; on the cloud pod -> /workspace)
tests/         fast CPU smoke tests
docs/          setup guides and the experiment log
```

## Getting started

- Laptop: [`docs/setup_local.md`](docs/setup_local.md)
- Pod: [`docs/setup_runpod.md`](docs/setup_runpod.md)

Then open [`notebooks/local/01_env_catalog.ipynb`](notebooks/local/01_env_catalog.ipynb),
which surveys every Playground environment and renders the robots. It is
committed with its outputs, so it is readable without running anything.

## What exists so far

| | |
|---|---|
| `envs/registry.py` | catalogue Playground's 54 envs and 26 robot models |
| `eval/rollout.py` | step an env, keep states / rewards / per-step metrics |
| `eval/render.py` | video, galleries, reward and action plots |
| `eval/figures.py` | high-resolution stills for the report |
| `scripts/view_model.py` | interactive MuJoCo viewer for composing figures |
| `docs/report/` | LaTeX survey of the locomotion suite — English (9 pp.) and Serbian Cyrillic (10 pp.) |

| `envs/erfi.py` | joystick task with ERFI torque perturbations; robot via `cfg.robot` (`go1` / `a1`), terrain via `cfg.task` (`flat_terrain` / `rough_terrain`) |
| `envs/a1/` | Unitree A1 port of Playground's Go1 joystick task (same names, Menagerie numbers); the paper's blind robot |
| `training/ppo.py` | Brax PPO wrapper; one self-describing directory per run |
| `eval/perturb.py` | robustness protocol of Campanaro et al. (payload, push, friction, gravity, Kp) |
| `scripts/train.py`, `scripts/eval.py` | study entry points, driven by `configs/experiment/*.yaml` |

## ERFI study

Flat terrain (done, results in `experiments/erfi_study_l2.5`):

```bash
python scripts/train.py --config configs/experiment/erfi_study.yaml
python scripts/eval.py  --config configs/experiment/erfi_study.yaml --plot
```

Rough terrain (same six conditions, Playground's 20 x 20 m heightfield, floor
friction 1.0 instead of 0.6):

```bash
python scripts/train.py --config configs/experiment/erfi_study_rough.yaml
python scripts/eval.py  --config configs/experiment/erfi_study_rough.yaml --plot
```

Unitree A1, the robot of the paper's blind experiment, with the same six
conditions and protocol:

```bash
python scripts/train.py --config configs/experiment/erfi_study_a1.yaml         # flat
python scripts/train.py --config configs/experiment/erfi_study_a1_rough.yaml   # rough
```

Robot and terrain are stored in each run's `env_config.json`, so evaluation
always rebuilds the model and scene the policy was trained on. Outputs land in
separate directories: `erfi_study_l2.5`, `erfi_study_rough_l2.5`,
`erfi_study_a1_l2.5`, `erfi_study_a1_rough_l2.5`. Run `pytest tests/test_erfi.py tests/test_a1.py`
(about four minutes on CPU) before sending anything to the pod.

## Common tasks

Browse the environments without opening a notebook:

```python
from rl_locomotion.envs import registry as reg

reg.print_models("locomotion")     # robots and their envs, as text
reg.envs_table("locomotion")       # timings, episode lengths, randomizers
reg.describe_env("Go1JoystickFlatTerrain")   # joint / actuator counts
```

Find a camera angle for a figure, then render it at full resolution:

```bash
mjpython scripts/view_model.py Go1JoystickFlatTerrain   # press P for the camera
```

```python
from rl_locomotion.eval import figures as fig
fig.save_model_figures(["Go1JoystickFlatTerrain"], out_dir="experiments/figures")
```

Build the locomotion-suite report:

```bash
cd docs/report && make          # both editions
make en                         # -> playground_locomotion.pdf     (English)
make sr                         # -> playground_lokomocija_sr.pdf  (српски, ћирилица)
make figures                    # re-render the robot plates first, if needed
make check-sr                   # catch Latin letters inside Cyrillic words
```

## How this is meant to be used

**Notebooks call the package; they don't contain the logic.** A training
notebook reads as *load config → `train(cfg)` → plot*. That keeps a good result
reproducible and lets the identical code run headless via `scripts/train.py`
for overnight jobs.

**One run = one immutable directory** under `experiments/runs/`, named by
timestamp and git SHA, containing the fully resolved config. Any figure can be
traced back to exactly what produced it.

**Configs override, they don't duplicate.** Playground already defines each
env's defaults; `configs/env/*.yaml` carries only the deltas, so upgrading
Playground doesn't silently desync your settings.

**Only `/workspace` persists on the pod.** `experiments/` is symlinked there.
Anything written elsewhere is gone when the pod stops.

## Two things that will bite you

**Rendering needs `MUJOCO_GL` set before MuJoCo is first imported** — `glfw` on
the laptop, `egl` on a headless pod. Setting it after the import silently has
no effect; the fix is to restart the kernel.

**Most locomotion rewards are clipped at zero.** Go1's joystick task computes
`reward = clip(sum(terms) * dt, 0, 10000)`, so a poor policy reports a flat
zero return while the underlying terms are strongly negative. Use
`Rollout.metrics_frame()` or `render.plot_reward_terms()` to see past the total.

This holds for 11 of the 15 locomotion tasks. G1 and Apollo joystick,
`h1/inplace_gait_tracking` and `spot/joystick_gait_tracking` do **not** clip and
can report negative returns — so the sign of a return means different things on
different robots. Check the task's `_get_reward` before comparing across them.
