#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

if ! command -v tmux >/dev/null 2>&1; then
    echo 'tmux is required but not found in PATH.' >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-${TIMESTAMP}}"
SESSION_NAME="${SESSION_NAME:-moe_end2end_${RUN_TAG}}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/end2end_bs64_maxnnz4/${RUN_TAG}}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
EA_TOTAL_TOKEN="${EA_TOTAL_TOKEN:-64}"
SPMLP_MAXNNZ="${SPMLP_MAXNNZ:-4}"
SPMLP_T_D="${SPMLP_T_D:-64}"
MTP_ADAPTIVE_TD="${MTP_ADAPTIVE_TD:-1}"
MTP_BM_FALLBACK_T_D="${MTP_BM_FALLBACK_T_D:--1}"
MTP_UNSUPPORTED_FALLBACK_MODE="${MTP_UNSUPPORTED_FALLBACK_MODE:-bm}"
MTP_RUNTIME_FALLBACK_MODE="${MTP_RUNTIME_FALLBACK_MODE:-none}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARY_ON_COMPLETE="${SUMMARY_ON_COMPLETE:-1}"
SUMMARY_POLL_INTERVAL="${SUMMARY_POLL_INTERVAL:-30}"

DATASET_SLUGS=(csqa gsm8k hellaswag piqa siqa sst2 alpaca sum)
GPU_IDS=(0 1 2 3 4 5 6 7)

mkdir -p "${RESULT_ROOT}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session ${SESSION_NAME} already exists." >&2
    exit 1
fi

{
    echo "session_name=${SESSION_NAME}"
    echo "result_root=${RESULT_ROOT}"
    echo "num_prompts=${NUM_PROMPTS}"
    echo "ea_total_token=${EA_TOTAL_TOKEN}"
    echo "spmlp_maxnnz=${SPMLP_MAXNNZ}"
    echo "spmlp_t_d=${SPMLP_T_D}"
    echo "mtp_adaptive_td=${MTP_ADAPTIVE_TD}"
    echo "mtp_bm_fallback_t_d=${MTP_BM_FALLBACK_T_D}"
    echo "mtp_unsupported_fallback_mode=${MTP_UNSUPPORTED_FALLBACK_MODE}"
    echo "mtp_runtime_fallback_mode=${MTP_RUNTIME_FALLBACK_MODE}"
    echo "dry_run=${DRY_RUN}"
    echo "summary_on_complete=${SUMMARY_ON_COMPLETE}"
    echo "summary_poll_interval=${SUMMARY_POLL_INTERVAL}"
    echo 'assignments:'
    for i in "${!DATASET_SLUGS[@]}"; do
        echo "  gpu${GPU_IDS[$i]}=${DATASET_SLUGS[$i]}"
    done
} > "${RESULT_ROOT}/launch_config.txt"

first_window="${DATASET_SLUGS[0]}"
tmux new-session -d -s "${SESSION_NAME}" -n "${first_window}"
tmux set-option -t "${SESSION_NAME}" remain-on-exit on

for i in "${!DATASET_SLUGS[@]}"; do
    dataset="${DATASET_SLUGS[$i]}"
    gpu_id="${GPU_IDS[$i]}"
    if [[ ${i} -gt 0 ]]; then
        tmux new-window -t "${SESSION_NAME}" -n "${dataset}"
    fi
    command="cd '${ROOT_DIR}' && NUM_PROMPTS='${NUM_PROMPTS}' EA_TOTAL_TOKEN='${EA_TOTAL_TOKEN}' SPMLP_MAXNNZ='${SPMLP_MAXNNZ}' SPMLP_T_D='${SPMLP_T_D}' MTP_ADAPTIVE_TD='${MTP_ADAPTIVE_TD}' MTP_BM_FALLBACK_T_D='${MTP_BM_FALLBACK_T_D}' MTP_UNSUPPORTED_FALLBACK_MODE='${MTP_UNSUPPORTED_FALLBACK_MODE}' MTP_RUNTIME_FALLBACK_MODE='${MTP_RUNTIME_FALLBACK_MODE}' DRY_RUN='${DRY_RUN}' '${ROOT_DIR}/run_one_dataset.sh' '${dataset}' '${gpu_id}' '${RESULT_ROOT}'"
    tmux send-keys -t "${SESSION_NAME}:${dataset}" "${command}" C-m
done

if [[ "$SUMMARY_ON_COMPLETE" == "1" ]]; then
    tmux new-window -t "$SESSION_NAME" -n summary
    summary_command="cd $ROOT_DIR && POLL_INTERVAL=$SUMMARY_POLL_INTERVAL $ROOT_DIR/watch_and_summarize.sh $RESULT_ROOT"
    tmux send-keys -t "$SESSION_NAME:summary" "$summary_command" C-m
fi

cat <<EOF
Started tmux session: ${SESSION_NAME}
Result root: ${RESULT_ROOT}
Attach with:
  tmux attach -t ${SESSION_NAME}
Summary output:
  ${RESULT_ROOT}/summary
Manual summarize command:
  ${ROOT_DIR}/summarize_end2end.py --result-root ${RESULT_ROOT}
EOF
