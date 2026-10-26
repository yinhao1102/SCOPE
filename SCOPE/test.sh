#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?Usage: bash test.sh /path/to/run-directory [checkpoint]}"
CHECKPOINT="${2:-R1_best.pth}"

ARGS=(
  --config_file "${RUN_DIR}/configs.yaml"
  --checkpoint "${CHECKPOINT}"
)

if [[ -n "${DATA_ROOT:-}" ]]; then
  ARGS+=(--root_dir "${DATA_ROOT}")
fi

python test.py "${ARGS[@]}"
