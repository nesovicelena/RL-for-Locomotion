#!/usr/bin/env bash
# Escape hatch: push uncommitted local code to the pod for tight iteration.
#
#   POD=root@1.2.3.4 POD_PORT=22 bash runpod/sync.sh
#
# Prefer `git push` + `git pull` on the pod for anything you intend to keep —
# code that only ever arrived by rsync is code you cannot reproduce later.
set -euo pipefail

POD="${POD:?set POD=user@host}"
POD_PORT="${POD_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/RL-for-Locomotion}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -avz --delete \
    -e "ssh -p ${POD_PORT}" \
    --exclude '.git/' \
    --exclude 'experiments/' \
    --exclude 'wandb/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '.ipynb_checkpoints/' \
    --exclude '.env' \
    "${LOCAL_DIR}/" "${POD}:${REMOTE_DIR}/"

echo "Synced ${LOCAL_DIR} -> ${POD}:${REMOTE_DIR}"
