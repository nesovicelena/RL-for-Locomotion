# RunPod setup

Pod-specific details (IPs, GPU choice, a log of what ran where) live in
[`runpod/notes.md`](../runpod/notes.md). This file is the procedure.

## 1. Create the pod

- CUDA 12 template (a stock PyTorch image is fine).
- **Attach a network volume mounted at `/workspace`.** Without it every
  checkpoint is lost when the pod stops.
- Size the GPU to the run: PPO throughput on MJX scales with `num_envs`, which
  is bounded by GPU memory.

## 2. Bootstrap

```bash
ssh -p <port> root@<ip>
cd /workspace && git clone https://github.com/nesovicelena/RL-for-Locomotion.git
cd RL-for-Locomotion && bash runpod/bootstrap.sh
```

Idempotent — re-run it on every fresh pod. It clones or pulls, installs,
symlinks `experiments/` to `/workspace/experiments`, and verifies JAX sees the
GPU.

## 3. Add secrets

Edit `.env` and set `WANDB_API_KEY`, then `wandb login`.

## 3b. Fetch the robot assets

Playground clones MuJoCo Menagerie (~2 GB) on first environment load. It lands
inside the Python environment, **not** on `/workspace`, so a fresh pod pays for
it again. Get it out of the way during setup rather than at the start of a
training run:

```bash
python -c "from rl_locomotion.envs.registry import ensure_menagerie; ensure_menagerie()"
```

## 4. Work

Short/interactive runs — Jupyter, tunnelled:

```bash
ssh -p <port> -L 8888:localhost:8888 root@<ip>
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Long runs — detach so the SSH connection dropping does not kill training:

```bash
tmux new -s train
python scripts/train.py --config configs/experiment/go1_ppo_baseline.yaml
# Ctrl-B D to detach
```

## 5. Get results back

Metrics and videos go to W&B automatically. For files:

```bash
POD=root@<ip> POD_PORT=<port> bash runpod/pull_results.sh go1_ppo_baseline
```

## 6. Stop the pod

Billing is hourly. A pod left running overnight after a run finished is the
most expensive mistake available here.

## Known failure modes

| Symptom | Cause |
|---|---|
| `jax.devices()` shows only CPU | JAX/CUDA version mismatch — reinstall from `requirements-gpu.txt` |
| GLFW / EGL error when rendering | `MUJOCO_GL` not `egl`, or set *after* MuJoCo was imported — restart the kernel |
| First env load hangs for minutes | cloning Menagerie (~2 GB); see step 3b |
| Checkpoints gone after restart | wrote outside `/workspace` |
| OOM at start of training | `num_envs` too large for this GPU |
| `scripts/view_model.py` fails | the interactive viewer needs a display; pods are headless, use `eval/figures.py` |
