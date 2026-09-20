#!/usr/bin/env bash
# One Spot study, end to end, on one pod: train, base protocol, all eight
# terrain suites. Three pods, one study each, in parallel:
#
#   STUDY=flat  STOP_POD=1 bash runpod/run_spot_everything.sh   # erfi_study_spot_v3
#   STUDY=rough STOP_POD=1 bash runpod/run_spot_everything.sh   # erfi_study_spot_v3_rough
#   STUDY=curr  STOP_POD=1 bash runpod/run_spot_everything.sh   # erfi_study_curr_spot_v3
#
# Six conditions x three seeds, v3 recipe (history_target_error only, no
# critic offset), Playground's Spot task kept whole, the mixin's 192-dim input
# on top (envs/spot/, envs/erfi.py). Same protocol and suites as Go1 and A1;
# payload and push as Go1's fractions of the robot's mass and weight
# (eval_terrain.ROBOT_ADJUST["spot"]).
#
# REFUSES TO START while the configs still carry the provisional torque limit.
# Run runpod/run_spot_limits.sh first (day 1), commit + push the measured
# vector, `git pull` here. ALLOW_PROVISIONAL=1 overrides, for smoke runs only.
#
# Restartable: finished runs, finished curriculum stages and evaluated
# policies are skipped, so the same command resumes after a pod restart.
#
# TIME (RTX 6000 Ada class; Spot has Go1's nv and 5 substeps, so Go1's rates):
#   flat   18 x ~9.3 min train + protocol ~20 min + suites 18 x 5.2 min  ->  ~4.5 h
#   rough  18 x ~11.4 min      + ~35 min          + ~1.6 h                ->  ~5.5 h
#   curr   18 x ~14.8 min      + ~35 min          + ~1.6 h                ->  ~6.5 h
#   Each fits one night. Spot is untested at scale: verify the first run's
#   `[none seed=0] step ...` lines in the log before leaving it unattended.
#
# Knobs (environment variables):
#   STUDY               flat | rough | curr                 (required)
#   RL_EXPERIMENTS_DIR  results root                       (default /workspace/experiments/redo)
#   N_EPISODES          episodes per protocol level        (default 50)
#   SMOKE=1             2 M-step pipeline check, then exit
#   SKIP_TRAIN=1 / SKIP_TERRAIN=1   skip a stage
#   ALLOW_PROVISIONAL=1 start even on the provisional limit (smoke only)
#   STOP_POD=1          `runpodctl stop pod` when finished (billing!)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

# Leave Warp room outside XLA's pool. JAX preallocates 75 % of the GPU by
# default, which is fine on 48 GB and not on 24 GB (RTX 4090: OOM in Brax's
# evaluator after a dozen ppo.train calls). 0.6 costs nothing on either.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}"

STUDY="${STUDY:?set STUDY=flat|rough|curr}"
case "${STUDY}" in
    flat)  CFG_NAME=erfi_study_spot_v3;       DIR=erfi_study_spot_v3;       CURR=0 ;;
    rough) CFG_NAME=erfi_study_spot_v3_rough; DIR=erfi_study_spot_v3_rough; CURR=0 ;;
    curr)  CFG_NAME=erfi_study_curr_spot_v3;  DIR=erfi_study_curr_spot_v3;  CURR=1 ;;
    *) echo "STUDY must be flat, rough or curr, got '${STUDY}'" >&2; exit 2 ;;
esac
CFG="configs/experiment/${CFG_NAME}.yaml"

if [ -z "${RL_EXPERIMENTS_DIR:-}" ]; then
    if [ -d /workspace ]; then RL_EXPERIMENTS_DIR=/workspace/experiments/redo; else RL_EXPERIMENTS_DIR="${REPO}/experiments/redo"; fi
fi
export RL_EXPERIMENTS_DIR
ROOT="${RL_EXPERIMENTS_DIR}"; LOGS="${ROOT}/logs"; STATUS="${ROOT}/status_spot_${STUDY}.txt"
mkdir -p "${LOGS}"
N_EPISODES="${N_EPISODES:-50}"

stamp() { date +%H:%M:%S; }
note() { echo "[$(stamp)] $*" | tee -a "${STATUS}"; }
count_finished() { ls "${ROOT}/$1"/*/seed*/params_final 2>/dev/null | wc -l | tr -d ' '; }
elapsed() { echo $(( ($(date +%s) - $1) / 60 )); }

# ---------------------------------------------------------------- guard
# The header keeps the word PROVISIONAL as documentation; the limit *lines* are
# what measure_stance_torque.py --write replaces, so key the guard on those.
if grep -Eq "^\s*(rfi|rao)_lim:.*PROVISIONAL" "${CFG}" && [ "${ALLOW_PROVISIONAL:-0}" != "1" ] && [ "${SMOKE:-0}" != "1" ]; then
    echo "REFUSING: ${CFG} still carries the PROVISIONAL torque limit." >&2
    echo "Run runpod/run_spot_limits.sh, commit + push the measured vector, git pull here." >&2
    echo "(ALLOW_PROVISIONAL=1 overrides; smoke runs only.)" >&2
    exit 3
fi

{
    echo "date    $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "commit  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "gpu     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"
    echo "study   ${STUDY} -> ${CFG_NAME}"; echo "limit   $(grep -h 'rfi_lim:' "${CFG}" | head -1 | sed 's/^ *//')"
    echo "dirty files:"; git status --short 2>/dev/null
} > "${ROOT}/provenance_spot_${STUDY}.txt"

T_ALL=$(date +%s)
note "==== run_spot_everything ${STUDY} start -> ${ROOT} (commit $(git rev-parse --short HEAD 2>/dev/null || echo ?))"
note "     ${DIR}: $(count_finished "${DIR}")/18 finished runs"

# ---------------------------------------------------------------- smoke
if [ "${SMOKE:-0}" = "1" ]; then
    if [ "${CURR}" = "1" ]; then
        python scripts/train_curriculum.py --config "${CFG}" --conditions none rfi --seeds 0 --smoke > "${LOGS}/smoke_spot_${STUDY}.log" 2>&1
    else
        STUDIES="${CFG_NAME}" SMOKE=1 bash runpod/run_all_studies.sh > "${LOGS}/smoke_spot_${STUDY}.log" 2>&1
    fi
    [ $? -eq 0 ] && note "DONE smoke ${STUDY}" || note "FAIL smoke ${STUDY} (see ${LOGS}/smoke_spot_${STUDY}.log)"
    exit 0
fi

# ---------------------------------------------------------------- 1 train + protocol
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    t0=$(date +%s); note "---- 1 train + protocol: ${CFG_NAME}"
    if [ "${CURR}" = "1" ]; then
        # One (condition, seed) per Python process. All 18 runs x 4 stages in one
        # process ran out of GPU memory on a 24 GB card after 16 ppo.train calls
        # (JAX keeps every compiled executable; Warp allocates outside XLA's
        # pool). A fresh process per run resets both; finished stages are
        # skipped, so this resumes exactly where a crash left off.
        rc=0
        for cond in none dr rfi rao erfi_c erfi_50; do
            for seed in 0 1 2; do
                if [ -d "${ROOT}/${DIR}/${cond}/seed${seed}/params_final" ]; then continue; fi
                note "     curriculum ${cond} seed ${seed}"
                python scripts/train_curriculum.py --config "${CFG}" --conditions "${cond}" --seeds "${seed}" \
                    >> "${LOGS}/${CFG_NAME}.log" 2>&1 || { rc=$?; note "FAIL curriculum ${cond} seed ${seed} exited ${rc}"; }
            done
        done
        if [ "$(count_finished "${DIR}")" = "18" ]; then
            python scripts/eval.py --config "${CFG}" --n-episodes "${N_EPISODES}" --plot >> "${LOGS}/${CFG_NAME}.log" 2>&1; rc=$?
        else
            rc=1
        fi
    else
        STUDIES="${CFG_NAME}" bash runpod/run_all_studies.sh > "${LOGS}/spot_${STUDY}_train.log" 2>&1; rc=$?
    fi
    [ ${rc} -ne 0 ] && note "FAIL 1 exited ${rc} after $(elapsed $t0) min (see ${LOGS})" || note "DONE 1 in $(elapsed $t0) min"
    note "     ${DIR}: $(count_finished "${DIR}")/18 finished runs"
fi

# ---------------------------------------------------------------- 2 terrain suites
if [ "$(count_finished "${DIR}")" != "18" ]; then
    note "SKIP 2 terrain suites: only $(count_finished "${DIR}")/18 policies finished. Re-run this command to resume training."
    SKIP_TERRAIN=1
fi
if [ "${SKIP_TERRAIN:-0}" != "1" ]; then
    t0=$(date +%s); note "---- 2 terrain suites on ${DIR}"
    python scripts/eval_terrain.py --studies "${DIR}" --n-episodes "${N_EPISODES}" >> "${LOGS}/spot_${STUDY}_terrain.log" 2>&1
    rc=$?
    [ ${rc} -ne 0 ] && note "FAIL 2 exited ${rc} after $(elapsed $t0) min (see ${LOGS}/spot_${STUDY}_terrain.log)" || note "DONE 2 in $(elapsed $t0) min"
fi

have=""
for s in "" _bowl_slope _rough_bowl_slope _rough_relief _rough_bowl_protocol _bowl_slope_fine _rough_bowl_slope_fine _combined_relief _combined_bowl; do
    [ -f "${ROOT}/${DIR}/results${s}.csv" ] && have="${have} ${s:-standard}"
done
note "     ${DIR}: $(count_finished "${DIR}")/18 runs, results:${have:- none}"
note "==== run_spot_everything ${STUDY} finished in $(elapsed $T_ALL) min; tree $(du -sh "${ROOT}" 2>/dev/null | cut -f1)"
echo; echo "Pull back with:"
echo "  rsync -avz --progress -e 'ssh -p <port>' --exclude 'params_0*' --exclude params_latest root@<ip>:${ROOT}/ experiments/redo/"

if [ "${STOP_POD:-0}" = "1" ]; then
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then note "stopping pod ${RUNPOD_POD_ID}"; runpodctl stop pod "${RUNPOD_POD_ID}"
    else note "STOP_POD=1 but runpodctl or RUNPOD_POD_ID missing; pod left running"; fi
fi
