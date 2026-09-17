#!/usr/bin/env bash
# The whole ERFI programme on Unitree A1, unattended: the Go1 redo tree
# reproduced on the paper's own blind robot (Campanaro et al. 2022, Sec. VI-B).
#
#   1  train      four studies x 6 conditions x 3 seeds x 300 M steps
#                   erfi_study_a1            v1, flat ground
#                   erfi_study_a1_rough      v1, rough terrain
#                   erfi_study_a1_v3         v3 (q* - q history), flat ground
#                   erfi_study_a1_v3_rough   v3, rough terrain
#                 each trained and then evaluated on the paper's five sweeps
#                 (payload, push, friction, gravity, Kp)
#
#   2  curriculum erfi_study_curr_a1_v3: relief curriculum, 4 stages x 75 M,
#                 each stage warm-started from the previous one, then the same
#                 five sweeps. A relief curriculum is a rough-terrain recipe by
#                 construction, so there is no flat counterpart -- the same as
#                 on Go1, where the five studies are v1 flat/rough, v3
#                 flat/rough and one curriculum.
#
#   3  terrain    bowl_slope, rough_bowl_slope, rough_relief, rough_bowl_protocol
#                 on all five A1 studies
#
#   4  extra      bowl_slope_fine, rough_bowl_slope_fine (10..26 deg, 16 s) and
#                 combined_relief, combined_bowl (payload x friction x push
#                 applied simultaneously) on all five A1 studies
#
# On the pod, inside tmux so a dropped SSH connection does not kill it:
#
#   tmux new -s a1
#   bash runpod/run_a1_everything.sh
#
# Everything lands beside the Go1 studies under $RL_EXPERIMENTS_DIR (default
# /workspace/experiments/redo), so one rsync brings both robots back and the
# two can be compared study by study.
#
# TIME: about 24 h end to end on one A100 -- roughly 13 h for stage 1, 4.5 h for
# stage 2, 3 h for stage 3 and 3.5 h for stage 4, extrapolated from the Go1 runs
# (18 runs of 300 M took 200 min; the Go1 curriculum took 264 min; four terrain
# suites over five studies took 168 min). Every stage skips work that is already
# done, so it is safe to stop it and re-run the same command later, and the
# stages can be run on separate days with the SKIP_* knobs.
#
# Knobs (environment variables):
#   RL_EXPERIMENTS_DIR  results root                     (default: see above)
#   NUM_TIMESTEPS       per-run budget for stage 1       (default 300000000, the Go1 budget)
#   NUM_EVALS           evaluations per run for stage 1  (default 15)
#   N_EPISODES          episodes per protocol level      (default 50, the paper's)
#   SMOKE=1             pipeline check of all four stages (minutes), then exit
#   SKIP_TRAIN=1 / SKIP_CURR=1 / SKIP_TERRAIN=1 / SKIP_EXTRA=1   skip a stage
#   STUDIES             stage 1 configs   (default: the four A1 training studies)
#   TERRAIN_STUDIES     stages 3-4 result dirs (default: the five A1 study dirs)
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
N_EPISODES="${N_EPISODES:-50}"
CURR_CFG=configs/experiment/erfi_study_curr_a1_v3.yaml
STUDIES="${STUDIES:-erfi_study_a1 erfi_study_a1_rough erfi_study_a1_v3 erfi_study_a1_v3_rough}"
TERRAIN_STUDIES="${TERRAIN_STUDIES:-erfi_study_a1_l2.5 erfi_study_a1_rough_l2.5 erfi_study_a1_v3_l2.5 erfi_study_a1_v3_rough_l2.5 erfi_study_curr_a1_v3_l2.5}"

stamp() { date +%H:%M:%S; }
note() { echo "[$(stamp)] $*" | tee -a "${STATUS}"; }
count_finished() { ls "${ROOT}/$1"/*/seed*/params_final 2>/dev/null | wc -l | tr -d ' '; }

# shellcheck disable=SC2086
suites_stage() {   # suites_stage <label> <logfile> <suite...>
    local label="$1" log="$2"; shift 2
    local t0; t0=$(date +%s)
    note "---- ${label}: suites $* on ${TERRAIN_STUDIES}"
    python scripts/eval_terrain.py --studies ${TERRAIN_STUDIES} --suites "$@" \
        --n-episodes "${N_EPISODES}" >> "${LOGS}/${log}" 2>&1
    local rc=$? dt
    dt=$(( ($(date +%s) - t0) / 60 ))
    if [ ${rc} -ne 0 ]; then
        note "FAIL ${label} exited ${rc} after ${dt} min (see ${LOGS}/${log})"
    else
        note "DONE ${label} in ${dt} min"
    fi
}

T_ALL=$(date +%s)
note "==== run_a1_everything start -> ${ROOT} (commit $(git rev-parse --short HEAD 2>/dev/null || echo ?))"
for s in ${TERRAIN_STUDIES}; do
    note "     ${s}: $(count_finished "${s}")/18 finished runs"
done

# ---------------------------------------------------------------- smoke
if [ "${SMOKE:-0}" = "1" ]; then
    note "---- smoke: every stage at minimum size"
    STUDIES="${STUDIES}" SMOKE=1 bash runpod/run_all_studies.sh \
        > "${LOGS}/smoke_a1_train.log" 2>&1 \
        && note "     stage 1 smoke ok" || note "FAIL stage 1 smoke (see ${LOGS}/smoke_a1_train.log)"
    python scripts/train_curriculum.py --config "${CURR_CFG}" --conditions none --seeds 0 --smoke \
        > "${LOGS}/smoke_a1_curr.log" 2>&1 \
        && note "     stage 2 smoke ok" || note "FAIL stage 2 smoke (see ${LOGS}/smoke_a1_curr.log)"
    # Stage 3/4 need a trained policy; use a Go1 study if the A1 ones are empty,
    # since the suites are robot-agnostic and this only checks the plumbing.
    probe=erfi_study_a1_v3_rough_l2.5
    [ "$(count_finished ${probe})" = "0" ] && probe=erfi_study_v3_rough_l2.5
    python scripts/eval_terrain.py --studies "${probe}" \
        --suites bowl_slope rough_relief bowl_slope_fine combined_bowl \
        --n-episodes 2 --force > "${LOGS}/smoke_a1_suites.log" 2>&1 \
        && note "     stages 3-4 smoke ok on ${probe}" || note "FAIL stages 3-4 smoke (see ${LOGS}/smoke_a1_suites.log)"
    rm -f "${ROOT}/${probe}"/results_bowl_slope.csv "${ROOT}/${probe}"/summary_*_bowl_slope.csv \
          "${ROOT}/${probe}"/results_rough_relief.csv "${ROOT}/${probe}"/summary_*_rough_relief.csv \
          "${ROOT}/${probe}"/results_bowl_slope_fine.csv "${ROOT}/${probe}"/summary_*_bowl_slope_fine.csv \
          "${ROOT}/${probe}"/results_combined_bowl.csv "${ROOT}/${probe}"/summary_*_combined_bowl.csv
    note "==== smoke done"
    exit 0
fi

# ---------------------------------------------------------------- 1. four studies
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    t0=$(date +%s)
    note "---- 1 train: ${STUDIES} at ${NUM_TIMESTEPS} steps"
    STUDIES="${STUDIES}" NUM_TIMESTEPS="${NUM_TIMESTEPS}" NUM_EVALS="${NUM_EVALS}" \
        bash runpod/run_all_studies.sh > "${LOGS}/a1_train.log" 2>&1
    rc=$?
    dt=$(( ($(date +%s) - t0) / 60 ))
    if [ ${rc} -ne 0 ]; then
        note "FAIL 1 train exited ${rc} after ${dt} min (see ${LOGS}/a1_train.log and logs/erfi_study_a1*.log)"
    else
        note "DONE 1 train in ${dt} min"
    fi
    for s in erfi_study_a1_l2.5 erfi_study_a1_rough_l2.5 erfi_study_a1_v3_l2.5 erfi_study_a1_v3_rough_l2.5; do
        note "     ${s}: $(count_finished "${s}")/18"
    done
fi

# ---------------------------------------------------------------- 2. curriculum
if [ "${SKIP_CURR:-0}" != "1" ]; then
    t0=$(date +%s)
    note "---- 2 curriculum a1 v3: $(count_finished erfi_study_curr_a1_v3_l2.5)/18 done, resuming"
    python scripts/train_curriculum.py --config "${CURR_CFG}" >> "${LOGS}/erfi_study_curr_a1_v3.log" 2>&1
    rc=$?
    if [ ${rc} -ne 0 ]; then
        note "FAIL 2 curriculum train exited ${rc} (see ${LOGS}/erfi_study_curr_a1_v3.log)"
    else
        python scripts/eval.py --config "${CURR_CFG}" --plot >> "${LOGS}/erfi_study_curr_a1_v3.log" 2>&1
        rc=$?
        dt=$(( ($(date +%s) - t0) / 60 ))
        if [ ${rc} -ne 0 ]; then
            note "FAIL 2 curriculum eval exited ${rc} after ${dt} min"
        else
            note "DONE 2 curriculum a1 v3 in ${dt} min"
        fi
    fi
fi

# ---------------------------------------------------------------- 3. terrain suites
[ "${SKIP_TERRAIN:-0}" != "1" ] && suites_stage "3 terrain suites" a1_eval_terrain.log \
    bowl_slope rough_bowl_slope rough_relief rough_bowl_protocol

# ---------------------------------------------------------------- 4. extra suites
[ "${SKIP_EXTRA:-0}" != "1" ] && suites_stage "4 extra suites" a1_eval_extra.log \
    bowl_slope_fine rough_bowl_slope_fine combined_relief combined_bowl

for s in ${TERRAIN_STUDIES}; do
    have=""
    for suite in "" _bowl_slope _rough_bowl_slope _rough_relief _rough_bowl_protocol \
                 _bowl_slope_fine _rough_bowl_slope_fine _combined_relief _combined_bowl; do
        [ -f "${ROOT}/${s}/results${suite}.csv" ] && have="${have} ${suite:-standard}"
    done
    note "     ${s}: $(count_finished "${s}")/18 runs, results:${have:- none}"
done

dt_all=$(( ($(date +%s) - T_ALL) / 60 ))
note "==== run_a1_everything finished in ${dt_all} min; tree $(du -sh "${ROOT}" 2>/dev/null | cut -f1)"
echo
echo "Pull back with:"
echo "  rsync -avz --progress -e 'ssh -p <port> -i ~/.ssh/id_ed25519' \\"
echo "    --exclude 'params_0*' --exclude params_latest \\"
echo "    root@<ip>:${ROOT}/ experiments/redo/"

if [ "${STOP_POD:-0}" = "1" ]; then
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
        note "stopping pod ${RUNPOD_POD_ID}"
        runpodctl stop pod "${RUNPOD_POD_ID}"
    else
        note "STOP_POD=1 but runpodctl or RUNPOD_POD_ID missing; pod left running"
    fi
fi
