#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [[ -x "${VENV_PYTHON}" ]]; then
    PYTHON_BIN="${VENV_PYTHON}"
else
    PYTHON_BIN="python"
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/moe_src${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_INPUT="${1:-}"
GPU_ID="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
RESULT_ROOT="${3:-${PROJECT_ROOT}/results/end2end_bs64_maxnnz4/manual}"

if [[ -z "${DATASET_INPUT}" ]]; then
    echo "Usage: $0 <dataset> [gpu_id] [result_root]" >&2
    exit 1
fi

NUM_PROMPTS="${NUM_PROMPTS:-100}"
EA_TOTAL_TOKEN="${EA_TOTAL_TOKEN:-64}"
SPMLP_MAXNNZ="${SPMLP_MAXNNZ:-4}"
SPMLP_T_D="${SPMLP_T_D:-64}"
MTP_ADAPTIVE_TD="${MTP_ADAPTIVE_TD:-1}"
MTP_BM_FALLBACK_T_D="${MTP_BM_FALLBACK_T_D:--1}"
MTP_UNSUPPORTED_FALLBACK_MODE="${MTP_UNSUPPORTED_FALLBACK_MODE:-bm}"
MTP_RUNTIME_FALLBACK_MODE="${MTP_RUNTIME_FALLBACK_MODE:-none}"
DRY_RUN="${DRY_RUN:-0}"

lower_dataset="$(printf '%s' "${DATASET_INPUT}" | tr '[:upper:]' '[:lower:]')"
case "${lower_dataset}" in
    csqa|commonsense_qa)
        DATASET_SLUG="csqa"
        DATASET_NAME="commonsense_qa"
        DATASET_DISPLAY="CSQA"
        DATASET_ID=1
        ;;
    gsm8k)
        DATASET_SLUG="gsm8k"
        DATASET_NAME="gsm8k"
        DATASET_DISPLAY="GSM8K"
        DATASET_ID=2
        ;;
    hellaswag)
        DATASET_SLUG="hellaswag"
        DATASET_NAME="hellaswag"
        DATASET_DISPLAY="HellaSwag"
        DATASET_ID=3
        ;;
    piqa)
        DATASET_SLUG="piqa"
        DATASET_NAME="piqa"
        DATASET_DISPLAY="PIQA"
        DATASET_ID=4
        ;;
    siqa)
        DATASET_SLUG="siqa"
        DATASET_NAME="siqa"
        DATASET_DISPLAY="SIQA"
        DATASET_ID=5
        ;;
    sst2|sst-2)
        DATASET_SLUG="sst2"
        DATASET_NAME="sst2"
        DATASET_DISPLAY="SST-2"
        DATASET_ID=6
        ;;
    alpaca)
        DATASET_SLUG="alpaca"
        DATASET_NAME="alpaca"
        DATASET_DISPLAY="Alpaca"
        DATASET_ID=0
        ;;
    sum)
        DATASET_SLUG="sum"
        DATASET_NAME="sum"
        DATASET_DISPLAY="SUM"
        DATASET_ID=7
        ;;
    *)
        echo "Unsupported dataset: ${DATASET_INPUT}" >&2
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

DATASET_RESULT_DIR="${RESULT_ROOT}/${DATASET_SLUG}"
mkdir -p "${DATASET_RESULT_DIR}"
RUN_LOG="${DATASET_RESULT_DIR}/driver.log"
STATUS_FILE="${DATASET_RESULT_DIR}/status.txt"

: > "${RUN_LOG}"
printf 'running\n' > "${STATUS_FILE}"

log() {
    local message="$1"
    printf '[%s][%s][gpu%s] %s\n' "$(date '+%F %T')" "${DATASET_DISPLAY}" "${GPU_ID}" "${message}" | tee -a "${RUN_LOG}"
}

build_command() {
    local method="$1"
    CMD=("${PYTHON_BIN}" -s main.py --dataset "${DATASET_ID}" --method "${method}" --num_prompts "${NUM_PROMPTS}")

    case "${method}" in
        hf)
            ;;
        eagle)
            CMD+=(--ea-total-token "${EA_TOTAL_TOKEN}")
            ;;
        mtp)
            CMD+=(
                --ea-total-token "${EA_TOTAL_TOKEN}"
                --spmlp-maxnnz "${SPMLP_MAXNNZ}"
                --spmlp-t-d "${SPMLP_T_D}"
            )
            if [[ "${MTP_ADAPTIVE_TD}" == "1" ]]; then
                CMD+=(--spmlp-adaptive-td)
            else
                CMD+=(--no-spmlp-adaptive-td)
            fi
            CMD+=(
                --spmlp-bm-fallback-td "${MTP_BM_FALLBACK_T_D}"
                --spmlp-unsupported-fallback-mode "${MTP_UNSUPPORTED_FALLBACK_MODE}"
                --spmlp-runtime-fallback-mode "${MTP_RUNTIME_FALLBACK_MODE}"
            )
            ;;
        bm|bmeagle)
            CMD+=(
                --ea-total-token "${EA_TOTAL_TOKEN}"
                --spmlp-maxnnz "${SPMLP_MAXNNZ}"
                --spmlp-t-d "${SPMLP_T_D}"
            )
            ;;
        *)
            echo "Unsupported method: ${method}" >&2
            exit 1
            ;;
    esac
}

run_method() {
    local method="$1"
    local method_log="${DATASET_RESULT_DIR}/${method}.log"

    build_command "${method}"
    log "starting method=${method}"

    {
        printf 'dataset_slug=%s\n' "${DATASET_SLUG}"
        printf 'dataset_name=%s\n' "${DATASET_NAME}"
        printf 'dataset_display=%s\n' "${DATASET_DISPLAY}"
        printf 'dataset_id=%s\n' "${DATASET_ID}"
        printf 'gpu_id=%s\n' "${GPU_ID}"
        printf 'num_prompts=%s\n' "${NUM_PROMPTS}"
        printf 'ea_total_token=%s\n' "${EA_TOTAL_TOKEN}"
        printf 'spmlp_maxnnz=%s\n' "${SPMLP_MAXNNZ}"
        printf 'spmlp_t_d=%s\n' "${SPMLP_T_D}"
        printf 'mtp_adaptive_td=%s\n' "${MTP_ADAPTIVE_TD}"
        printf 'mtp_bm_fallback_t_d=%s\n' "${MTP_BM_FALLBACK_T_D}"
        printf 'mtp_unsupported_fallback_mode=%s\n' "${MTP_UNSUPPORTED_FALLBACK_MODE}"
        printf 'mtp_runtime_fallback_mode=%s\n' "${MTP_RUNTIME_FALLBACK_MODE}"
        printf 'method=%s\n' "${method}"
        printf 'command=%s\n' "$(printf '%q ' "${CMD[@]}")"
        if [[ "${DRY_RUN}" == "1" ]]; then
            echo '[dry-run] command not executed'
        else
            "${CMD[@]}"
        fi
    } 2>&1 | tee "${method_log}"
    local rc=${PIPESTATUS[0]}

    printf 'exit_code=%s\n' "${rc}" >> "${method_log}"
    log "finished method=${method} exit_code=${rc} log=${method_log}"

    if [[ ${rc} -ne 0 ]]; then
        printf 'failed\n' > "${STATUS_FILE}"
        return ${rc}
    fi
}

cd "${ROOT_DIR}"
log "suite start result_dir=${DATASET_RESULT_DIR} num_prompts=${NUM_PROMPTS} ea_total_token=${EA_TOTAL_TOKEN} maxnnz=${SPMLP_MAXNNZ} t_d=${SPMLP_T_D} dry_run=${DRY_RUN}"

for method in hf eagle mtp bm bmeagle; do
    run_method "${method}"
done

printf 'success\n' > "${STATUS_FILE}"
log 'suite completed successfully'
