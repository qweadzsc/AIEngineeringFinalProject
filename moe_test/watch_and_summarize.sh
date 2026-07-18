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

RESULT_ROOT="${1:-}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
DATASET_SLUGS=(csqa gsm8k hellaswag piqa siqa sst2 alpaca sum)

if [[ -z "${RESULT_ROOT}" ]]; then
    echo "Usage: $0 <result_root>" >&2
    exit 1
fi

is_terminal_status() {
    local status="$1"
    [[ "${status}" == "success" || "${status}" == "failed" ]]
}

echo "[watch_and_summarize] waiting for dataset runs under ${RESULT_ROOT}"
echo "[watch_and_summarize] poll interval: ${POLL_INTERVAL}s"

while true; do
    all_done=1
    for dataset in "${DATASET_SLUGS[@]}"; do
        status_file="${RESULT_ROOT}/${dataset}/status.txt"
        if [[ ! -f "${status_file}" ]]; then
            all_done=0
            break
        fi
        status="$(tr -d '[:space:]' < "${status_file}")"
        if ! is_terminal_status "${status}"; then
            all_done=0
            break
        fi
    done

    if [[ ${all_done} -eq 1 ]]; then
        break
    fi

    sleep "${POLL_INTERVAL}"
done

echo "[watch_and_summarize] all dataset runs reached terminal state, generating summary"
"${PYTHON_BIN}" -s "${ROOT_DIR}/summarize_end2end.py" --result-root "${RESULT_ROOT}" | tee "${RESULT_ROOT}/summary_generation.log"
echo "[watch_and_summarize] done"
