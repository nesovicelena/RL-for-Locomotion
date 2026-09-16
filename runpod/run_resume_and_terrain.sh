#!/usr/bin/env bash
# Resume after the pod loss, then the Go1 terrain programme, unattended.
#
#   1  v3 rough      finish erfi_study_v3_rough through run_all_studies.sh
#                    (skips the runs already in the redo tree; trains + evaluates)
#   2  curriculum    terrain curriculum with the v3 observation
#                    (scripts/train_curriculum.py, 4 x 75 M) + the standard protocol
#   3  terrain       scripts/eval_terrain.py on every study present:
#                    bowl_slope, rough_bowl_slope, rough_relief, rough_bowl_protocol
#
# On the pod, inside tmux so a dropped SSH connection does not kill it:
#
#   tmux new -s all
#   bash runpod/run_resume_and_terrain.sh
#
# Expects the pulled results at $RL_EXPERIMENTS_DIR (default /workspace/experiments/redo):
# the five finished studies plus whatever exists of erfi_study_v3_rough. Every
# stage skips what is already done, so re-running after an interruption resumes.
# A failing stage is logged in status.txt and the next one starts.
#
# Knobs (environment variables):
#   RL_EXPERIMENTS_DIR  results root                    (default: see above)
#   NUM_TIMESTEPS       per-run budget for stage 1       (default 300000000, the redo budget)
#   NUM_EVALS           evaluations per run for stage 1  (default 15)
#   SMOKE=1             pipeline check: curriculum --smoke and a 2-episode terrain suite, then exit
#   SKIP_V3_ROUGH=1 / SKIP_CURR=1 / SKIP_TERRAIN=1   skip a stage
#   TERRAIN_STUDIES     studies for stage 3 (default: the five Go1 studies + the curriculum)
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

NUM_TIMESTEPS="${NUM_TIMESTEPS:-300000000}"
NUM_EVALS="${NUM_EVALS:-15}"
CURR_CFG=configs/experiment/erfi_study_curr_v3.yaml
TERRAIN_STUDIES="${TERRAIN_STUDIES:-erfi_study_l2.5 erfi_study_rough_l2.5 erfi_study_v3_l2.5 erfi_study_v3_rough_l2.5 erfi_study_curr_v3_l2.5}"

stamp() { date +%H:%M:%S; }
note() { echo "[$(stamp)] $*" | tee -a "${STATUS}"; }
count_finished() { ls "${ROOT}/$1"/*/seed*/params_final 2>/dev/null | wc -l | tr -d ' '; }

T_ALL=$(date +%s)
note "==== run_resume_and_terrain start -> ${ROOT} (commit $(git rev-parse --short HEAD 2>/dev/null || echo ?))"
for s in erfi_study_l2.5 erfi_study_rough_l2.5 erfi_study_v2_l2.5 erfi_study_v2_rough_l2.5 erfi_study_v3_l2.5 erfi_study_v3_rough_l2.5; do
    note "     found ${s}: $(count_finished "${s}")/18 finished runs"
done

# ---------------------------------------------------------------- smoke
if [ "${SMOKE:-0}" = "1" ]; then
    note "---- smoke: curriculum (2 stages x 1 M) and a 2-episode terrain suite"
    python scripts/train_curriculum.py --config "${CURR_CFG}" --conditions none --seeds 0 --smoke \
        > "${LOGS}/smoke_curriculum.log" 2>&1 && note "     curriculum smoke ok" || note "FAIL curriculum smoke (see ${LOGS}/smoke_curriculum.log)"
    python scripts/eval_terrain.py --studies erfi_study_l2.5 --suites bowl_slope --n-episodes 2 --no-plot --force \
        > "${LOGS}/smoke_terrain.log" 2>&1 && note "     terrain smoke ok" || note "FAIL terrain smoke (see ${LOGS}/smoke_terrain.log)"
    # the smoke suite wrote a 2-episode results_bowl_slope.csv into the v1 flat study; remove it
    rm -f "${ROOT}/erfi_study_l2.5/results_bowl_slope.csv" "${ROOT}/erfi_study_l2.5/summary_success_rate_bowl_slope.csv"
    note "==== smoke done"
    exit 0
fi

# ---------------------------------------------------------------- 1. v3 rough
if [ "${SKIP_V3_ROUGH:-0}" != "1" ]; then
    t0=$(date +%s)
    note "---- 1 v3 rough: $(count_finished erfi_study_v3_rough_l2.5)/18 done, resuming"
    STUDIES="erfi_study_v3_rough" NUM_TIMESTEPS="${NUM_TIMESTEPS}" NUM_EVALS="${NUM_EVALS}" \
        bash runpod/run_all_studies.sh > "${LOGS}/resume_v3_rough.log" 2>&1
    rc=$?
    dt=$(( ($(date +%s) - t0) / 60 ))
    if [ ${rc} -ne 0 ] || [ ! -f "${ROOT}/erfi_study_v3_rough_l2.5/results.csv" ]; then
        note "FAIL 1 v3 rough after ${dt} min (see ${LOGS}/resume_v3_rough.log and logs/erfi_study_v3_rough.log)"
    else
        note "DONE 1 v3 rough in ${dt} min"
    fi
fi

# ---------------------------------------------------------------- 2. curriculum v3
if [ "${SKIP_CURR:-0}" != "1" ]; then
    t0=$(date +%s)
    note "---- 2 curriculum v3: $(count_finished erfi_study_curr_v3_l2.5)/18 done, resuming"
    python scripts/train_curriculum.py --config "${CURR_CFG}" >> "${LOGS}/erfi_study_curr_v3.log" 2>&1
    rc=$?
    if [ ${rc} -ne 0 ]; then
        note "FAIL 2 curriculum train exited ${rc} (see ${LOGS}/erfi_study_curr_v3.log)"
    else
        python scripts/eval.py --config "${CURR_CFG}" --plot >> "${LOGS}/erfi_study_curr_v3.log" 2>&1
        rc=$?
        dt=$(( ($(date +%s) - t0) / 60 ))
        if [ ${rc} -ne 0 ]; then
            note "FAIL 2 curriculum eval exited ${rc} after ${dt} min"
        else
            note "DONE 2 curriculum v3 in ${dt} min"
        fi
    fi
fi

# ---------------------------------------------------------------- 3. terrain suites
if [ "${SKIP_TERRAIN:-0}" != "1" ]; then
    t0=$(date +%s)
    note "---- 3 terrain suites on: ${TERRAIN_STUDIES}"
    # shellcheck disable=SC2086
    python scripts/eval_terrain.py --studies ${TERRAIN_STUDIES} >> "${LOGS}/eval_terrain.log" 2>&1
    rc=$?
    dt=$(( ($(date +%s) - t0) / 60 ))
    if [ ${rc} -ne 0 ]; then
        note "FAIL 3 terrain suites exited ${rc} after ${dt} min (see ${LOGS}/eval_terrain.log)"
    else
        note "DONE 3 terrain suites in ${dt} min"
    fi
fi

dt_all=$(( ($(date +%s) - T_ALL) / 60 ))
note "==== run_resume_and_terrain finished in ${dt_all} min; tree $(du -sh "${ROOT}" 2>/dev/null | cut -f1)"
echo
echo "Pull back with:"
echo "  rsync -avz --progress -e 'ssh -p <port> -i ~/.ssh/id_ed25519' --exclude 'params_0*' --exclude params_latest root@<ip>:${ROOT}/ experiments/redo/"

if [ "${STOP_POD:-0}" = "1" ]; then
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
        note "stopping pod ${RUNPOD_POD_ID}"
        runpodctl stop pod "${RUNPOD_POD_ID}"
    else
        note "STOP_POD=1 but runpodctl or RUNPOD_POD_ID missing; pod left running"
    fi
fi
