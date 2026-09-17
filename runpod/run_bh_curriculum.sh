#!/usr/bin/env bash
# Berkeley Humanoid: terrain suites on the finished studies, then the terrain
# curriculum, then its evaluation. Unattended and restartable.
#
#   1  suites      terrain suites on erfi_study_bh_rough and erfi_study_bh   (~1 h each)
#   2  smoke       curriculum pipeline check, 2 stages x 1 M steps            (a few min)
#   3  curriculum  18 runs x 4 stages of 50 M                                 (~8-9 h)
#   4  protocol    the paper's sweeps on the curriculum runs                  (~30 min)
#   5  suites_curr terrain suites on the curriculum runs                      (~1 h)
#
# On the pod, inside tmux so a dropped SSH connection does not kill it:
#
#   tmux new -s bh2
#   bash runpod/run_bh_curriculum.sh
#
# Stage 1 needs the two finished studies under RL_EXPERIMENTS_DIR; it reports and
# continues if they are absent. Stages 3 to 5 depend on each other and stop on a
# hard failure. Everything resumes: finished curriculum stages are skipped
# individually, and each terrain suite skips runs it has already evaluated.
#
# Knobs (environment variables):
#   RL_EXPERIMENTS_DIR  output root       (default /workspace/experiments/redo, or ./experiments/redo)
#   SMOKE=1             run stage 2 only, then exit
#   SUITES=0            skip stage 1 (the suites on the finished studies)
#   CURRICULUM=0        skip stages 2 to 5 (suites only)
#   CONDITIONS          conditions for stage 3     (default: the config's six)
#   SEEDS               seeds for stage 3          (default: the config's 0 1 2)
#   ONLY_SUITES         terrain suites to run      (default: all four)
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
STATUS="${ROOT}/status_bh_curr.txt"
mkdir -p "${LOGS}"

CFG_CURR=configs/experiment/erfi_study_curr_bh.yaml
OUT_CURR="${ROOT}/erfi_study_curr_bh"
DONE_STUDIES="erfi_study_bh_rough erfi_study_bh"
SUITE_FLAG=""
if [ -n "${ONLY_SUITES:-}" ]; then
    SUITE_FLAG="--suites ${ONLY_SUITES}"
fi

stamp() { date +%H:%M:%S; }
note() { echo "[$(stamp)] $*" | tee -a "${STATUS}"; }
run() {  # run <log-name> <command...>; logs to LOGS/<log-name>.log, returns the exit code
    local log="${LOGS}/$1.log"; shift
    echo "[$(stamp)] \$ $*" >> "${log}"
    "$@" >> "${log}" 2>&1
}

if [ ! -f "${CFG_CURR}" ]; then
    echo "ERROR: ${CFG_CURR} not found. The pod is on the wrong branch or has not pulled." >&2
    echo "       git log --oneline -1   should show the curriculum commit." >&2
    exit 1
fi

{
    echo "date    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "branch  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "gpu     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"
    echo "suites ${SUITES:-1}  curriculum ${CURRICULUM:-1}  only_suites ${ONLY_SUITES:-<all>}"
    echo "dirty files:"; git status --short 2>/dev/null
} > "${ROOT}/provenance_bh_curr.txt"

T_ALL=$(date +%s)
note "==== run_bh_curriculum start -> ${ROOT}"

# ---------------------------------------------------------------- 2 smoke only
if [ "${SMOKE:-0}" = "1" ]; then
    note "---- stage 2: curriculum smoke (none, erfi_50; seed 0; 2 stages x 1 M)"
    # shellcheck disable=SC2086
    run curr_smoke_bh python scripts/train_curriculum.py --config "${CFG_CURR}" \
        --conditions none erfi_50 --seeds 0 --smoke
    rc=$?
    if [ ${rc} -eq 0 ]; then
        note "DONE smoke -> ${OUT_CURR}_smoke (check stages/s0_a0.000 and s1_a0.015 per run)"
    else
        note "FAIL smoke: exit ${rc} (see ${LOGS}/curr_smoke_bh.log)"
    fi
    exit ${rc}
fi

# ---------------------------------------------------------------- 1 suites on the finished studies
if [ "${SUITES:-1}" = "1" ]; then
    present=""
    for s in ${DONE_STUDIES}; do
        if [ -d "${ROOT}/${s}" ]; then present="${present} ${s}"; else note "stage 1: ${s} not under ${ROOT}, skipping it"; fi
    done
    if [ -n "${present}" ]; then
        note "---- stage 1: terrain suites on${present}"
        # shellcheck disable=SC2086
        run terrain_bh python scripts/eval_terrain.py --studies ${present} ${SUITE_FLAG}
        rc=$?
        if [ ${rc} -ne 0 ]; then
            note "FAIL stage 1: exit ${rc} (see ${LOGS}/terrain_bh.log); continuing to the curriculum"
        else
            note "DONE stage 1 -> ${ROOT}/<study>/results_<suite>.csv"
        fi
    else
        note "stage 1: no finished humanoid studies found, skipping"
    fi
fi

if [ "${CURRICULUM:-1}" != "1" ]; then
    note "CURRICULUM=0, stopping after the suites"
else

# ---------------------------------------------------------------- 3 curriculum
note "---- stage 3: terrain curriculum (4 stages x 50 M per run)"
CURR_FLAGS=""
if [ -n "${CONDITIONS:-}" ]; then CURR_FLAGS="${CURR_FLAGS} --conditions ${CONDITIONS}"; fi
if [ -n "${SEEDS:-}" ]; then CURR_FLAGS="${CURR_FLAGS} --seeds ${SEEDS}"; fi
# shellcheck disable=SC2086
run curr_bh python scripts/train_curriculum.py --config "${CFG_CURR}" ${CURR_FLAGS}
if [ $? -ne 0 ]; then
    note "FAIL stage 3: training exited non-zero (see ${LOGS}/curr_bh.log); stopping"
    exit 1
fi
finished=$(find "${OUT_CURR}" -maxdepth 3 -name params_final 2>/dev/null | wc -l | tr -d ' ')
note "DONE stage 3: ${finished} finished runs under ${OUT_CURR}"

# ---------------------------------------------------------------- 4 protocol
note "---- stage 4: protocol on the curriculum runs"
run curr_eval_bh python scripts/eval.py --config "${CFG_CURR}" --plot
if [ $? -ne 0 ]; then
    note "FAIL stage 4: eval exited non-zero (see ${LOGS}/curr_eval_bh.log); continuing to the suites"
else
    note "DONE stage 4 -> ${OUT_CURR}/results.csv"
fi

# ---------------------------------------------------------------- 5 suites on the curriculum
note "---- stage 5: terrain suites on the curriculum runs"
# shellcheck disable=SC2086
run terrain_curr_bh python scripts/eval_terrain.py --studies erfi_study_curr_bh ${SUITE_FLAG}
if [ $? -ne 0 ]; then
    note "FAIL stage 5: exit $? (see ${LOGS}/terrain_curr_bh.log)"
else
    note "DONE stage 5 -> ${OUT_CURR}/results_<suite>.csv"
fi

fi  # CURRICULUM

dt_all=$(( ($(date +%s) - T_ALL) / 60 ))
note "==== run_bh_curriculum finished in ${dt_all} min"
note "tree size: $(du -sh "${ROOT}" 2>/dev/null | cut -f1)"
echo
echo "Pull it back to the laptop with:"
echo "  rsync -avz --progress --exclude 'params_0*' -e 'ssh -p <port>' \\"
echo "    'root@<ip>:${ROOT}/{erfi_study_curr_bh,erfi_study_bh*,status_bh_curr.txt,logs}' experiments/redo/"

if [ "${STOP_POD:-0}" = "1" ]; then
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
        note "stopping pod ${RUNPOD_POD_ID}"
        runpodctl stop pod "${RUNPOD_POD_ID}"
    else
        note "STOP_POD=1 but runpodctl or RUNPOD_POD_ID missing; pod left running"
    fi
fi
