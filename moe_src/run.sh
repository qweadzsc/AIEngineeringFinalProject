#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_SPEED_TEST=0
PASSTHROUGH_ARGS=()

while (($# > 0)); do
    case "$1" in
        --speed)
            RUN_SPEED_TEST=1
            ;;
        -h|--help)
            cat <<'EOF'
Usage: bash moe_src/run.sh [--speed]

Options:
  --speed    Enable dense-only Triton speed benchmarks and parameter sweep.
EOF
            exit 0
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            ;;
    esac
    shift
done

if [[ "${RUN_SPEED_TEST}" == "1" ]]; then
    export RUN_MIXED_SDDMM_BENCH="${RUN_MIXED_SDDMM_BENCH:-1}"
    export RUN_MIXED_SDDMM_TUNE="${RUN_MIXED_SDDMM_TUNE:-1}"
    echo "[run] speed mode enabled: RUN_MIXED_SDDMM_BENCH=${RUN_MIXED_SDDMM_BENCH} RUN_MIXED_SDDMM_TUNE=${RUN_MIXED_SDDMM_TUNE}"
fi

bash "$SCRIPT_DIR/run_test.sh" "${PASSTHROUGH_ARGS[@]}"
