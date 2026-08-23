#!/usr/bin/env bash
# Bring one run's artifacts back to the laptop for figure-making.
#
#   POD=root@1.2.3.4 bash runpod/pull_results.sh go1_ppo_baseline
#
# Day-to-day metrics should come from W&B instead; use this when you need the
# actual files (a checkpoint to re-render, raw metrics for a thesis plot).
set -euo pipefail

POD="${POD:?set POD=user@host}"
POD_PORT="${POD_PORT:-22}"
RUN="${1:?usage: pull_results.sh <experiment-name>}"
REMOTE_RUNS="${REMOTE_RUNS:-/workspace/experiments/runs}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/experiments/runs"

mkdir -p "${LOCAL_DIR}"

# Checkpoints are large and stay on the volume unless asked for explicitly.
EXCLUDE=(--exclude 'checkpoints/')
if [ "${WITH_CHECKPOINTS:-0}" = "1" ]; then
    EXCLUDE=()
fi

rsync -avz --progress \
    -e "ssh -p ${POD_PORT}" \
    "${EXCLUDE[@]}" \
    "${POD}:${REMOTE_RUNS}/${RUN}/" "${LOCAL_DIR}/${RUN}/"

echo "Pulled ${RUN} -> ${LOCAL_DIR}/${RUN}"
echo "(set WITH_CHECKPOINTS=1 to include checkpoint files)"
