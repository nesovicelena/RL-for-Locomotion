#!/usr/bin/env bash
# RFI at the physics rate: the three ERFI conditions retrained with
# `erfi.per_substep` and put through every evaluation, on one robot.
# One pod per robot, in parallel:
#
#   ROBOT=go1 bash runpod/run_substep_everything.sh
#   ROBOT=a1  bash runpod/run_substep_everything.sh
#   ROBOT=bh  bash runpod/run_substep_everything.sh
#
# Why (envs/erfi.py, module docstring): every run in experiments/redo draws
# tau_r once per control step, 50 Hz, because that is the only injection point
# above Playground's substep loop. On Go1 that is 0.62 of the knee's 32 ms time
# constant, so the joint partially tracks each draw and RFI behaves partly like
# a short RAO. `erfi.per_substep` redraws inside the loop: 250 Hz on the
# quadrupeds, 500 Hz on the humanoid.
#
# Only rfi / erfi_c / erfi_50 are trained: none, dr and rao are bit-identical
# with the option on or off (tests/test_erfi.py), so their existing policies
# stand and are compared against these. Per robot: v3 flat, v3 rough and the
# v3 terrain curriculum, three seeds each -> 27 policies, into
# <study>_substep directories beside the originals.
#
#   1  train      flat and rough studies through run_all_studies.sh (which also
#                 runs the base protocol on each when done)
#   2  curriculum train_curriculum.py, then the base protocol
#   3  terrain    all eight suites on the three new directories
#
# On the pod, inside tmux so a dropped SSH connection does not kill it:
#
#   tmux new -s substep
#   ROBOT=go1 STOP_POD=1 bash runpod/run_substep_everything.sh
#
# Restartable: finished runs, finished curriculum stages and evaluated
# policies are all skipped, so the same command resumes after a pod restart.
#
# TIME (from the redo tree's own logs, RTX 6000 Ada class):
#   go1 / a1   train 9 x (9.3 + 11.4 + 14.8) min ~ 5.3 h, protocol ~1 h,
#              suites 27 x 5.2 min ~ 2.3 h                     -> about 8.5 h
#   bh         train 9 x (26 + 26 + 30) min ~ 12.3 h, protocol 27 x 4.7 min
#              ~ 2.1 h, suites 27 x 8.5 min ~ 3.8 h            -> about 18 h
#   The humanoid does not fit one night; SKIP_TERRAIN=1 brings it to ~14.5 h
#   and the suites can be run the next day with the same command (SKIP_TRAIN=1
#   SKIP_CURR=1), since everything resumes.
#
# Knobs (environment variables):
#   ROBOT               go1 | a1 | bh                      (required)
#   RL_EXPERIMENTS_DIR  results root                       (default /workspace/experiments/redo)
#   N_EPISODES          episodes per protocol level        (default 50)
#   SMOKE=1             2 M-step pipeline check of the three studies, then exit
#   SKIP_TRAIN=1 / SKIP_CURR=1 / SKIP_TERRAIN=1   skip a stage
#   STOP_POD=1          `runpodctl stop pod` when finished (billing!)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

ROBOT="${ROBOT:?set ROBOT=go1|a1|bh}"
case "${ROBOT}" in
    go1) FLAT=erfi_study_v3_substep;    ROUGH=erfi_study_v3_rough_substep;    CURR=erfi_study_curr_v3_substep
         DIRS="erfi_study_v3_substep_l2.5 erfi_study_v3_rough_substep_l2.5 erfi_study_curr_v3_substep_l2.5" ;;
    a1)  FLAT=erfi_study_a1_v3_substep; ROUGH=erfi_study_a1_v3_rough_substep; CURR=erfi_study_curr_a1_v3_substep
         DIRS="erfi_study_a1_v3_substep_l2.5 erfi_study_a1_v3_rough_substep_l2.5 erfi_study_curr_a1_v3_substep_l2.5" ;;
    bh)  FLAT=erfi_study_bh_substep;    ROUGH=erfi_study_bh_rough_substep;    CURR=erfi_study_curr_bh_substep
         DIRS="erfi_study_bh_substep erfi_study_bh_rough_substep erfi_study_curr_bh_substep" ;;
    *)   echo "ROBOT must be go1, a1 or bh, got '${ROBOT}'" >&2; exit 2 ;;
esac

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
STATUS="${ROOT}/status_substep_${ROBOT}.txt"
mkdir -p "${LOGS}"
N_EPISODES="${N_EPISODES:-50}"
CURR_CFG="configs/experiment/${CURR}.yaml"

stamp() { date +%H:%M:%S; }
note() { echo "[$(stamp)] $*" | tee -a "${STATUS}"; }
count_finished() { ls "${ROOT}/$1"/*/seed*/params_final 2>/dev/null | wc -l | tr -d ' '; }
elapsed() { echo $(( ($(date +%s) - $1) / 60 )); }

{
    echo "date    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "gpu     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"
    echo "robot   ${ROBOT}   studies ${FLAT} ${ROUGH} ${CURR}"
    echo "dirty files:"; git status --short 2>/dev/null
} > "${ROOT}/provenance_substep_${ROBOT}.txt"

T_ALL=$(date +%s)
note "==== run_substep_everything ${ROBOT} start -> ${ROOT} (commit $(git rev-parse --short HEAD 2>/dev/null || echo ?))"
for d in ${DIRS}; do note "     ${d}: $(count_finished "${d}")/9 finished runs"; done

# ---------------------------------------------------------------- smoke
if [ "${SMOKE:-0}" = "1" ]; then
    STUDIES="${FLAT} ${ROUGH}" SMOKE=1 bash runpod/run_all_studies.sh > "${LOGS}/smoke_substep_${ROBOT}.log" 2>&1 \
        && note "     smoke flat/rough ok" || note "FAIL smoke flat/rough (see ${LOGS}/smoke_substep_${ROBOT}.log)"
    python scripts/train_curriculum.py --config "${CURR_CFG}" --conditions rfi --seeds 0 --smoke \
        > "${LOGS}/smoke_substep_curr_${ROBOT}.log" 2>&1 \
        && note "     smoke curriculum ok" || note "FAIL smoke curriculum (see ${LOGS}/smoke_substep_curr_${ROBOT}.log)"
    note "==== smoke done"; exit 0
fi

# ---------------------------------------------------------------- 1. flat + rough
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    t0=$(date +%s); note "---- 1 train+protocol: ${FLAT} ${ROUGH}"
    STUDIES="${FLAT} ${ROUGH}" bash runpod/run_all_studies.sh > "${LOGS}/substep_${ROBOT}_train.log" 2>&1
    rc=$?
    [ ${rc} -ne 0 ] && note "FAIL 1 exited ${rc} after $(elapsed $t0) min (see ${LOGS}/substep_${ROBOT}_train.log)" \
                    || note "DONE 1 in $(elapsed $t0) min"
fi

# ---------------------------------------------------------------- 2. curriculum
if [ "${SKIP_CURR:-0}" != "1" ]; then
    t0=$(date +%s); note "---- 2 curriculum: ${CURR}"
    python scripts/train_curriculum.py --config "${CURR_CFG}" >> "${LOGS}/${CURR}.log" 2>&1
    rc=$?
    if [ ${rc} -ne 0 ]; then
        note "FAIL 2 curriculum train exited ${rc} (see ${LOGS}/${CURR}.log)"
    else
        python scripts/eval.py --config "${CURR_CFG}" --n-episodes "${N_EPISODES}" --plot >> "${LOGS}/${CURR}.log" 2>&1
        rc=$?
        [ ${rc} -ne 0 ] && note "FAIL 2 curriculum eval exited ${rc} after $(elapsed $t0) min" \
                        || note "DONE 2 in $(elapsed $t0) min"
    fi
fi

# ---------------------------------------------------------------- 3. terrain suites
if [ "${SKIP_TERRAIN:-0}" != "1" ]; then
    t0=$(date +%s); note "---- 3 terrain suites on ${DIRS}"
    # shellcheck disable=SC2086
    python scripts/eval_terrain.py --studies ${DIRS} --n-episodes "${N_EPISODES}" >> "${LOGS}/substep_${ROBOT}_terrain.log" 2>&1
    rc=$?
    [ ${rc} -ne 0 ] && note "FAIL 3 exited ${rc} after $(elapsed $t0) min (see ${LOGS}/substep_${ROBOT}_terrain.log)" \
                    || note "DONE 3 in $(elapsed $t0) min"
fi

for d in ${DIRS}; do
    have=""
    for s in "" _bowl_slope _rough_bowl_slope _rough_relief _rough_bowl_protocol \
             _bowl_slope_fine _rough_bowl_slope_fine _combined_relief _combined_bowl; do
        [ -f "${ROOT}/${d}/results${s}.csv" ] && have="${have} ${s:-standard}"
    done
    note "     ${d}: $(count_finished "${d}")/9 runs, results:${have:- none}"
done
note "==== run_substep_everything ${ROBOT} finished in $(elapsed $T_ALL) min; tree $(du -sh "${ROOT}" 2>/dev/null | cut -f1)"
echo; echo "Pull back with:"
echo "  rsync -avz --progress -e 'ssh -p <port>' --exclude 'params_0*' --exclude params_latest \\"
echo "    root@<ip>:${ROOT}/ experiments/redo/"

if [ "${STOP_POD:-0}" = "1" ]; then
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
        note "stopping pod ${RUNPOD_POD_ID}"; runpodctl stop pod "${RUNPOD_POD_ID}"
    else
        note "STOP_POD=1 but runpodctl or RUNPOD_POD_ID missing; pod left running"
    fi
fi
