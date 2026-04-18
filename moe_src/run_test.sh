#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

DEFAULT_TESTS=(
    "test/test_mixed_sddmm_metadata.py"
    "test/test_mixed_sddmm_reference.py"
)

GPU_TESTS=()
LEGACY_OPTIONAL_TESTS=(
    "test/test_oss.py"
)

run_python_test() {
    local test_path="$1"
    echo "[run_test] python ${test_path}"
    python "${test_path}"
}

contains_path() {
    local needle="$1"
    shift

    local candidate
    for candidate in "$@"; do
        if [[ "${candidate}" == "${needle}" ]]; then
            return 0
        fi
    done
    return 1
}

discover_tests() {
    local test_path
    while IFS= read -r test_path; do
        if contains_path "${test_path}" "${DEFAULT_TESTS[@]}"; then
            continue
        fi
        if contains_path "${test_path}" "${LEGACY_OPTIONAL_TESTS[@]}"; then
            continue
        fi
        GPU_TESTS+=("${test_path}")
    done < <(find test -maxdepth 1 -type f -name 'test*.py' | sort)
}

run_test_group() {
    local group_name="$1"
    shift

    local test_path
    for test_path in "$@"; do
        echo "[run_test] ${group_name}: ${test_path}"
        run_python_test "${test_path}"
    done
}

build_extension() {
    echo "[run_test] python setup.py build_ext --inplace"
    python setup.py build_ext --inplace
}

discover_tests
run_test_group "DEFAULT" "${DEFAULT_TESTS[@]}"
printf '[run_test] Skipping legacy non-default test: %s\n' "${LEGACY_OPTIONAL_TESTS[@]}"

if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    echo "[run_test] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    build_extension
    run_test_group "GPU" "${GPU_TESTS[@]}"
else
    echo "[run_test] CUDA is not available. Skipping GPU-backed tests:"
    printf '[run_test]   %s\n' "${GPU_TESTS[@]}"
fi
