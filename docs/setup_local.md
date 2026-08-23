# Local setup (laptop)

The laptop is for writing code, editing configs, running tests, and making plots from finished runs. **No training happens here** — MJX on a Mac is CPU only and locomotion PPO would take days.

## Requirements

We use conda.

## Install

```bash
conda env create -f environment.yml
conda activate rl-locomotion
pip install -r requirements-cpu.txt
pip install -e ".[dev]"
cp .env.example .env      # set MUJOCO_GL=glfw locally
```

Conda supplies only the interpreter; everything else comes from pip against
`pyproject.toml`, so the laptop and the pod resolve the same dependency set.

## Check it works

```bash
python -c "import jax, mujoco_playground; print(jax.devices())"
pytest                                    # fast CPU smoke tests
jupyter lab notebooks/local/
```

Expect `[CpuDevice(id=0)]` — that is correct here. A GPU device would be a
surprise; training belongs on the pod.

## Jupyter kernel

JupyterLab launched from inside the activated env picks it up automatically.
To reach it from another Jupyter install:

```bash
python -m ipykernel install --user --name rl-locomotion
```

If your editor reports "package not installed" on `import rl_locomotion`, it is
pointing at a different interpreter — select the `rl-locomotion` env as the
kernel.

## Robot assets (MuJoCo Menagerie)

Playground ships the task XMLs but not the robot meshes. Those come from MuJoCo
Menagerie, cloned on first environment load — **about 2 GB**, once, into the
conda env rather than the repo.

```python
from rl_locomotion.envs import registry as reg
reg.menagerie_available()   # False until fetched
reg.ensure_menagerie()      # download
```

Cataloguing environments works without it; building or rendering a model does
not.

## Rendering

Set `MUJOCO_GL` **before MuJoCo is first imported** — `glfw` here, `egl` on the
headless pod. Setting it afterwards silently does nothing, and the symptom is an
opaque GL error; the fix is to restart the kernel.

```python
import os
os.environ.setdefault("MUJOCO_GL", "glfw")   # first cell, before other imports
```

The interactive viewer needs `mjpython`, not `python` — on macOS it must own the
main thread:

```bash
mjpython scripts/view_model.py Go1JoystickFlatTerrain
```

## What runs where

| Task | Machine |
|---|---|
| Editing `src/`, configs | laptop |
| `pytest` | laptop |
| Browsing envs, stills, short rollouts, video | laptop |
| Interactive MuJoCo viewer | laptop only (the pod has no display) |
| **Training** | **pod** |
| Long or many-episode evaluation | pod |
| Reading W&B, thesis plots | laptop |

Exploration and rendering are perfectly usable on CPU — a 100-step Go1 rollout
takes about ten seconds and rendering it about one. It is *training* that is
hopeless here: PPO needs thousands of parallel environments.

`notebooks/local/` assumes no GPU. `notebooks/remote/` assumes one — opening
those here will be slow or fail, and that is intentional.
