#!/usr/bin/env bash
# Keep pulling the redo tree from one or more pods, on a timer, until Ctrl-C.
#
#   PODS="root@1.2.3.4:22001 root@5.6.7.8:22002" bash runpod/pull_loop.sh
#   PODS="root@1.2.3.4:22001" INTERVAL=10 bash runpod/pull_loop.sh        # every 10 s
#   PODS="..." FULL=1 bash runpod/pull_loop.sh                            # include params_final
#
# Light mode (default) brings status/provenance files, logs, curves, run
# configs, results CSVs and figures -- everything text -- and skips every
# checkpoint. FULL=1 adds params_final (2-3 MB per policy) but still skips the
# intermediate params_0* checkpoints. rsync is incremental, so a quiet pod costs
# one SSH round trip per pass. A pod that is unreachable (stopped, rebooting)
# is reported and skipped; the loop goes on.
#
# After each pass the last status line of every study is printed, so this
# window doubles as the progress monitor. Run under `caffeinate -i` if the
# laptop may sleep. Ctrl-C to stop.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PODS="${PODS:?set PODS=\"user@host:port user@host:port ...\"}"
INTERVAL="${INTERVAL:-60}"
REMOTE="${REMOTE:-/workspace/experiments/redo}"
LOCAL="${REPO}/experiments/redo"
mkdir -p "${LOCAL}"

FILTER=(--exclude 'params_0*' --exclude 'params_latest'
        --include='*/' --include='status_*.txt' --include='provenance_*.txt' --include='logs/*.log'
        --include='curve.json' --include='summary.json' --include='spec.json' --include='env_config.json'
        --include='ppo_config.json' --include='stance_torque.json' --include='results*.csv'
        --include='summary_*.csv' --include='*.png')
if [ "${FULL:-0}" = "1" ]; then
    FILTER+=(--include='params_final')
else
    FILTER+=(--exclude 'params_final')
fi
FILTER+=(--exclude='*')

pass=0
while true; do
    pass=$((pass + 1))
    for pod in ${PODS}; do
        host="${pod%%:*}"; port="${pod##*:}"; [ "${port}" = "${pod}" ] && port=22
        if rsync -az --timeout=30 -e "ssh -p ${port} -o ConnectTimeout=10 -o BatchMode=yes" \
                "${FILTER[@]}" "${host}:${REMOTE}/" "${LOCAL}/" 2>/dev/null; then
            printf '[%s] pass %d  %-28s ok\n' "$(date +%H:%M:%S)" "${pass}" "${pod}"
        else
            printf '[%s] pass %d  %-28s UNREACHABLE (stopped? wrong port?)\n' "$(date +%H:%M:%S)" "${pass}" "${pod}"
        fi
    done
    for f in "${LOCAL}"/status_spot_*.txt; do
        [ -f "${f}" ] && printf '    %-24s %s\n' "$(basename "${f}" .txt | sed 's/status_//')" "$(tail -1 "${f}")"
    done
    for d in erfi_study_spot_v3 erfi_study_spot_v3_rough erfi_study_curr_spot_v3; do
        n=$(ls "${LOCAL}/${d}"/*/seed*/summary.json 2>/dev/null | wc -l | tr -d ' ')
        r=$(ls "${LOCAL}/${d}"/results*.csv 2>/dev/null | wc -l | tr -d ' ')
        [ -d "${LOCAL}/${d}" ] && printf '    %-28s %2s/18 politika, %s results*.csv\n' "${d}" "${n}" "${r}"
    done
    sleep "${INTERVAL}"
done
