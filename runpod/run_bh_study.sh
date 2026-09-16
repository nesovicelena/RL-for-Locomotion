#!/usr/bin/env bash
# ERFI study on the Berkeley Humanoid, rough terrain then flat, unattended.
# Follows docs/humanoid_design.md Section 9 step by step:
#
#   0  smoke        2 M-step pipeline check (only with SMOKE=1; exits afterwards)
#   1  timing       one 20 M-step `none` run; prints the 200 M estimate   (skip: TIMING=0)
#   2  none         `none` x seeds on rough terrain, 200 M each
#   3  measure      stance-torque vector from the best-walking `none` seed,
#                   written into both bh configs (skipped once a config already
#                   carries a measured vector)
#   4  sweep        0.25 / 0.5 / 1.0 x that vector, erfi_50 + rao, seed 0, 50 M  (skip: SWEEP=0)
#   5  study        the two full studies through run_all_studies.sh:
#                   erfi_study_bh_rough (15 remaining runs + eval), erfi_study_bh (18 + eval)
#
# On the pod, inside tmux so a dropped SSH connection does not kill it:
#
#   tmux new -s bh
#   bash runpod/run_bh_study.sh
#
# Restartable: every stage skips what already exists (finished runs have a
# params_final; the measured vector is recognised by its provenance comment).
#
# Knobs (environment variables):
#   RL_EXPERIMENTS_DIR  output root          (default /workspace/experiments/redo, or ./experiments/redo)
#   SMOKE=1             run stage 0 only
#   TIMING=0            skip stage 1
#   SWEEP=0             skip stage 4
#   FRACTION            limit = FRACTION x RMS actuator force          (default 0.5, the paper's "half a stance torque")
#   SEEDS               seeds for stage 2 (default: the config's; "0 1 2")
#   NUM_TIMESTEPS       override the configs' 200 M per run for stages 2 and 5
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
STATUS="${ROOT}/status_bh.txt"
mkdir -p "${LOGS}"

CFG_ROUGH=configs/experiment/erfi_study_bh_rough.yaml
CFG_FLAT=configs/experiment/erfi_study_bh.yaml
OUT_ROUGH="${ROOT}/erfi_study_bh_rough"
FRACTION="${FRACTION:-0.5}"
TRAIN_FLAGS=""
if [ -n "${NUM_TIMESTEPS:-}" ]; then
    TRAIN_FLAGS="--num-timesteps ${NUM_TIMESTEPS}"
fi

stamp() { date +%H:%M:%S; }
note() { echo "[$(stamp)] $*" | tee -a "${STATUS}"; }
run() {  # run <log-name> <command...>; logs to LOGS/<log-name>.log, returns the exit code
    local log="${LOGS}/$1.log"; shift
    echo "[$(stamp)] \$ $*" >> "${log}"
    "$@" >> "${log}" 2>&1
}

{
    echo "date    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "branch  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "gpu     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"
    echo "fraction ${FRACTION}  timing ${TIMING:-1}  sweep ${SWEEP:-1}  train flags ${TRAIN_FLAGS:-<config>}"
    echo "dirty files:"; git status --short 2>/dev/null
} > "${ROOT}/provenance_bh.txt"

T_ALL=$(date +%s)
note "==== run_bh_study start -> ${ROOT}"

# ---------------------------------------------------------------- 0 smoke
if [ "${SMOKE:-0}" = "1" ]; then
    note "---- stage 0: smoke (none, erfi_50; seed 0; 2 M steps; rough)"
    run smoke_bh python scripts/train.py --config "${CFG_ROUGH}" --conditions none erfi_50 --seeds 0 --smoke \
        && run smoke_bh python scripts/eval.py --config "${CFG_ROUGH}" --runs "${OUT_ROUGH}_smoke" --n-episodes 2 --plot
    rc=$?
    if [ ${rc} -eq 0 ]; then
        note "DONE smoke; check ${OUT_ROUGH}_smoke/results.csv for push_N_sagittal/push_N_lateral rows and a low_base_rate column"
    else
        note "FAIL smoke: exit ${rc} (see ${LOGS}/smoke_bh.log)"
    fi
    exit ${rc}
fi

# ---------------------------------------------------------------- 1 timing
if [ "${TIMING:-1}" = "1" ]; then
    TIMING_RUN="${ROOT}/erfi_study_bh_rough_timing/none/seed0"
    if [ -f "${TIMING_RUN}/params_final" ]; then
        note "skip stage 1: timing run exists"
    else
        note "---- stage 1: timing (none seed 0, 20 M steps, 2 evals)"
        run timing_bh python scripts/train.py --config "${CFG_ROUGH}" --conditions none --seeds 0 \
            --num-timesteps 20000000 --num-evals 2 --out erfi_study_bh_rough_timing
        [ $? -ne 0 ] && note "FAIL timing (see ${LOGS}/timing_bh.log); continuing"
    fi
    if [ -f "${TIMING_RUN}/summary.json" ]; then
        est=$(python - "${TIMING_RUN}" <<'EOF'
import json, sys
r = sys.argv[1]
c = json.load(open(f"{r}/curve.json")); s = json.load(open(f"{r}/summary.json"))
jit, total = c[0]["wall_s"], s["wall_s"]
print(f"20 M in {total/60:.1f} min (JIT {jit/60:.1f}) -> 200 M about {(total - jit) * 10 / 60:.0f} min per run")
EOF
)
        note "timing: ${est}"
    fi
fi

# ---------------------------------------------------------------- 2 none seeds (rough)
note "---- stage 2: none seeds on rough terrain"
if [ -n "${SEEDS:-}" ]; then
    # shellcheck disable=SC2086
    run none_bh python scripts/train.py --config "${CFG_ROUGH}" --conditions none --seeds ${SEEDS} ${TRAIN_FLAGS}
else
    # shellcheck disable=SC2086
    run none_bh python scripts/train.py --config "${CFG_ROUGH}" --conditions none ${TRAIN_FLAGS}
fi
if [ $? -ne 0 ]; then
    note "FAIL stage 2: train exited non-zero (see ${LOGS}/none_bh.log); stopping"
    exit 1
fi
note "DONE stage 2"

# ---------------------------------------------------------------- 3 stance-torque vector
if grep -q "rfi_lim:.*measured" "${CFG_ROUGH}"; then
    note "skip stage 3: ${CFG_ROUGH} already carries a measured vector"
else
    note "---- stage 3: nominal eval of none, pick the best walker, measure stance torques"
    run measure_bh python scripts/eval.py --config "${CFG_ROUGH}" --params payload_kg --force
    best=$(python - "${OUT_ROUGH}/results.csv" <<'EOF'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
nom = df[(df.condition == "none") & (df.param == "payload_kg") & (df.level == 0.0)]
row = nom.sort_values("progress_m", ascending=False).iloc[0]
print(row["run"], f"{row['progress_m']:.2f}")
EOF
)
    best_run=$(echo "${best}" | cut -d' ' -f1); best_prog=$(echo "${best}" | cut -d' ' -f2)
    note "best none policy: ${best_run} (${best_prog} m at nominal)"
    if python -c "import sys; sys.exit(0 if float('${best_prog}') >= 2.5 else 1)"; then
        run measure_bh python scripts/measure_stance_torque.py "${OUT_ROUGH}/${best_run}" \
            --fraction "${FRACTION}" --write "${CFG_ROUGH}" "${CFG_FLAT}"
        if [ $? -ne 0 ]; then
            note "FAIL stage 3: measurement failed (see ${LOGS}/measure_bh.log); stopping"
            exit 1
        fi
        note "DONE stage 3: limits written to both configs (git diff configs/experiment/)"
        grep -h "rfi_lim:" "${CFG_ROUGH}" | tee -a "${STATUS}"
    else
        note "FAIL stage 3: no none seed walks 2.5 m at nominal (best ${best_prog} m). Provisional limits kept; stopping."
        exit 1
    fi
    # The partial results (payload only, none only) would make eval.py skip these
    # runs later; the full study evaluates everything in one pass.
    rm -f "${OUT_ROUGH}/results.csv" "${OUT_ROUGH}/summary_success_rate.csv"
fi

# ---------------------------------------------------------------- 4 limit sweep
if [ "${SWEEP:-1}" = "1" ]; then
    note "---- stage 4: limit sweep 0.25 / 0.5 / 1.0 x vector (erfi_50, rao; seed 0; 50 M)"
    for f in 0.25 0.5 1.0; do
        LIM=$(python -c "import yaml,sys; f=float(sys.argv[1]); v=yaml.safe_load(open('${CFG_ROUGH}'))['train']['rfi_lim']; print(' '.join(f'{f*x:.3f}' for x in v))" "${f}")
        # shellcheck disable=SC2086
        run sweep_bh python scripts/train.py --config "${CFG_ROUGH}" --conditions erfi_50 rao --seeds 0 \
            --num-timesteps 50000000 --num-evals 5 --rfi-lim ${LIM} --rao-lim ${LIM} --out "erfi_limits_bh_f${f}"
        [ $? -ne 0 ] && note "FAIL sweep f=${f} train (see ${LOGS}/sweep_bh.log)"
        run sweep_bh python scripts/eval.py --config "${CFG_ROUGH}" --runs "${ROOT}/erfi_limits_bh_f${f}" --params payload_kg gravity
    done
    note "sweep summary (final eval reward, nominal progress):"
    python - "${ROOT}" <<'EOF' | tee -a "${STATUS}"
import json, sys, glob, os
import pandas as pd
root = sys.argv[1]
for f in ("0.25", "0.5", "1.0"):
    d = f"{root}/erfi_limits_bh_f{f}"
    res = pd.read_csv(f"{d}/results.csv") if os.path.exists(f"{d}/results.csv") else None
    for run in sorted(glob.glob(f"{d}/*/seed0")):
        cond = os.path.basename(os.path.dirname(run))
        s = json.load(open(f"{run}/summary.json")) if os.path.exists(f"{run}/summary.json") else {}
        prog = "n/a"
        if res is not None:
            nom = res[(res.condition == cond) & (res.param == "payload_kg") & (res.level == 0.0)]
            if len(nom):
                prog = f"{float(nom.progress_m.iloc[0]):.2f} m"
        print(f"  f={f:5s} {cond:8s} reward {s.get('final_reward', float('nan')):7.2f}  nominal progress {prog}")
EOF
    note "decision rule (design 4.1): largest fraction at which erfi_50 and rao leave the standing plateau and erfi_50 walks 2.5 m."
    note "if it is not ${FRACTION}, rerun stage 3 with FRACTION=<f> after deleting the measured lines, or edit the configs by hand."
fi

# ---------------------------------------------------------------- 5 full studies
note "---- stage 5: full studies (rough, then flat) through run_all_studies.sh"
STUDIES="erfi_study_bh_rough erfi_study_bh" NUM_TIMESTEPS="${NUM_TIMESTEPS:-}" bash runpod/run_all_studies.sh
rc=$?

dt_all=$(( ($(date +%s) - T_ALL) / 60 ))
note "==== run_bh_study finished in ${dt_all} min (run_all_studies exit ${rc})"
echo
echo "Pull it back to the laptop with:"
echo "  rsync -avz --progress --exclude 'params_0*' -e 'ssh -p <port>' root@<ip>:${ROOT}/ experiments/redo/"
echo "and commit the measured limits: git add configs/experiment/erfi_study_bh*.yaml"

if [ "${STOP_POD:-0}" = "1" ]; then
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
        note "stopping pod ${RUNPOD_POD_ID}"
        runpodctl stop pod "${RUNPOD_POD_ID}"
    else
        note "STOP_POD=1 but runpodctl or RUNPOD_POD_ID missing; pod left running"
    fi
fi
