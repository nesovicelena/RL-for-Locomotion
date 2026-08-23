# Docker

Not in use yet. The current workflow is a stock CUDA 12 RunPod template plus
`runpod/bootstrap.sh`, which is faster to get started with.

Build the image once the dependency set stops changing — at that point the
bootstrap script's `pip install` step is pure waiting, repeated on every new
pod, and baking it into an image removes it.

When that happens:

1. Fill in `Dockerfile` (CUDA 12 base, Python 3.11, `requirements-gpu.txt`).
2. `docker build -t <user>/rl-locomotion:<tag> docker/`
3. Push, then point a RunPod custom template at it.
4. Reduce `bootstrap.sh` to: pull repo, symlink `experiments/`, launch Jupyter.
