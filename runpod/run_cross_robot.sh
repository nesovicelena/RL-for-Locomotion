#!/usr/bin/env bash
# Zero-shot cross-robot evaluation: every v3 Go1 policy on the A1 model and
# every v3 A1 policy on the Go1 model, under the paper's protocol. Evaluation
# only: nothing is trained, no checkpoint is written. The v1 studies are left
# out on purpose (v3 is the recipe the thesis reports); pass GO1_STUDIES /
# A1_STUDIES to include them.
#
# Why. Go1 and A1 share the task code, the 192-dim state, the 12 actions, the
# default pose and Kp/Kd, so a checkpoint of one runs on the other unchanged
# (scripts/eval.py --robot). What differs is dynamics the policy never saw:
# trunk mass 5.20 vs 4.71 kg, leg 0.213 vs 0.200 m, foot radius 0.023 vs
# 0.020 m, actuator limits 23.7/35.55 vs 33.5 Nm. That is the "shift in mass,
# inertia and kinematics" RAO claims to absorb implicitly (paper Sec. V-B),
# tested by swapping the robot instead of one parameter, and the closest this
# setup gets to the paper's cross-simulator deployment.
#
# On the pod, inside tmux so a dropped SSH connection does not kill it:
#
#   tmux new -s cross
#   bash runpod/run_cross_robot.sh
#
# Writes <study>/results_on_<robot>.csv, summary_success_rate_on_<robot>.csv
# and the three curve PNGs next to each study's own results.csv. Evaluated
# runs are skipped, so re-running after an interruption resumes.
# Afterwards: python scripts/transfer_table.py  (laptop, no GPU).
#
# Time: the full protocol of one study (18 policies, 35 levels) took about
# 20 min on the A1 pod runs, so six studies are roughly 2 h on the pod; measured
# on an M3 Pro CPU it is about 3 h (flat 1.7 s and rough 3.4 s per level of
# 50 episodes). QUICK=1 keeps payload and push only (pod ~40 min, Mac ~1.5 h).
#
# Knobs (environment variables):
#   RL_EXPERIMENTS_DIR  results root                     (default /workspace/experiments/redo)
#   N_EPISODES          episodes per level               (default 50, the protocol's)
#   QUICK=1             --params payload_kg push_N only
#   GO1_STUDIES / A1_STUDIES   config names to evaluate  (default: the three v3 of each)
#   STOP_POD=1          `runpodctl stop pod` when finished (billing!)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

if [ -z "${RL_EXPERIMENTS_DIR:-}" ]; then
    if [ -d /workspace ]; then
        RL_EXPERIMENTS_DIR=/workspace/experiments/redo
    else
        RL_EXPERIMENTS_DIR="${REPO}/experiments/redo"
    fi
fi
export RL_EXPERIMENTS_DIR
ROOT="${RL_EXPERIMENTS_DIR}"
LOGS="${ROOT}/logs"
STATUS="${ROOT}/status.txt"
mkdir -p "${LOGS}"

N_EPISODES="${N_EPISODES:-50}"
GO1_STUDIES="${GO1_STUDIES:-erfi_study_v3 erfi_study_v3_rough erfi_study_curr_v3}"
A1_STUDIES="${A1_STUDIES:-erfi_study_a1_v3 erfi_study_a1_v3_rough erfi_study_curr_a1_v3}"
PARAM_FLAGS=""
if [ "${QUICK:-0}" = "1" ]; then
    PARAM_FLAGS="--params payload_kg push_N"
fi

stamp() { date +%H:%M:%S; }
note() { echo "[$(stamp)] $*" | tee -a "${STATUS}"; }

# shellcheck disable=SC2086
evaluate_on() {   # evaluate_on <target robot> <config name...>
    local robot="$1"; shift
    for study in "$@"; do
        local cfg="configs/experiment/${study}.yaml" log="${LOGS}/cross_${study}_on_${robot}.log"
        if [ ! -f "${cfg}" ]; then
            note "SKIP ${study}: no such config ${cfg}"
            continue
        fi
        local t0; t0=$(date +%s)
        note "---- ${study} -> ${robot}"
        python scripts/eval.py --config "${cfg}" --robot "${robot}" --n-episodes "${N_EPISODES}" \
            ${PARAM_FLAGS} --plot >> "${log}" 2>&1
        local rc=$? dt
        dt=$(( ($(date +%s) - t0) / 60 ))
        if [ ${rc} -ne 0 ]; then
            note "FAIL ${study} -> ${robot}: exited ${rc} after ${dt} min (see ${log})"
        else
            note "DONE ${study} -> ${robot} in ${dt} min"
        fi
    done
}

T_ALL=$(date +%s)
note "==== run_cross_robot start -> ${ROOT} (commit $(git rev-parse --short HEAD 2>/dev/null || echo ?))"
note "     Go1 studies on A1: ${GO1_STUDIES}"
note "     A1 studies on Go1: ${A1_STUDIES}"
[ -n "${PARAM_FLAGS}" ] && note "     QUICK: ${PARAM_FLAGS}"

evaluate_on a1  ${GO1_STUDIES}
evaluate_on go1 ${A1_STUDIES}

dt_all=$(( ($(date +%s) - T_ALL) / 60 ))
note "==== run_cross_robot finished in ${dt_all} min"
echo
echo "Pull back with:"
echo "  rsync -avz --progress -e 'ssh -p <port>' --include='*/' --include='results_on_*.csv' \\"
echo "    --include='summary_success_rate_on_*.csv' --include='*_curves_on_*.png' --exclude='*' \\"
echo "    root@<ip>:${ROOT}/ experiments/redo/"
echo "then:  python scripts/transfer_table.py"

if [ "${STOP_POD:-0}" = "1" ]; then
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
        note "stopping pod ${RUNPOD_POD_ID}"
        runpodctl stop pod "${RUNPOD_POD_ID}"
    else
        note "STOP_POD=1 but runpodctl or RUNPOD_POD_ID missing; pod left running"
    fi
fi
