#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

run_python_test() {
    local test_path="$1"
    echo "[run_test] python ${test_path}"
    python "${test_path}"
}

build_extension() {
    echo "[run_test] python setup.py build_ext --inplace"
    python setup.py build_ext --inplace
}

run_python_test "test/test_mixed_sddmm_metadata.py"

if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    echo "[run_test] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    build_extension
    run_python_test "test/test.py"
else
    echo "[run_test] CUDA is not available. Skipping GPU-backed tests:"
    echo "[run_test]   test/test.py"
fi
