# RunPod notes

Working notes about the pods themselves. Keep this current — it is what saves
you when you come back after two weeks and have to recreate everything.

## Current pod

| | |
|---|---|
| GPU | _e.g. 1x A100 80GB_ |
| Template | _e.g. runpod/pytorch:2.4.0-py3.11-cuda12.4.1_ |
| Volume | `/workspace`, _N_ GB |
| Cost | _$/hr_ |

## Connect

```bash
export POD=root@<ip>
export POD_PORT=<port>
ssh -p $POD_PORT $POD
```

Jupyter (tunnel it rather than exposing the port publicly):

```bash
ssh -p $POD_PORT -L 8888:localhost:8888 $POD
# on the pod:
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## First run on a new pod

```bash
bash runpod/bootstrap.sh
```

## Gotchas

- **Only `/workspace` survives a pod stop, and only if it is a network volume.**
  A pod without one lost the whole `redo` tree on 2026-09-16. Attach a network
  volume at `/workspace` when creating the pod, and rsync results back after
  every finished study, not at the end.
- **`MUJOCO_GL=egl`** for headless rendering. Without it, video rendering fails
  with an obscure GLFW error.
- **Stop the pod when idle.** Billing is per-hour and a forgotten pod is the
  most expensive bug in this repo.
- Record the working `pip freeze` into `requirements-gpu.txt` after the first
  successful training run.

## Pod log

| Date | GPU | What was run | Outcome |
|---|---|---|---|
| 2026-09-15/16 | RTX 4090 | `run_all_studies.sh` (v1, v2, v3 x flat/rough, 300 M) | 5 studies finished and pulled; pod evicted during v3 rough (2/18). No network volume, disk lost. |
| 2026-09-16 | | `run_resume_and_terrain.sh` on a new pod | resume v3 rough, curriculum v3, terrain suites |
