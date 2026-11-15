#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command="${1:-train}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
  train) exec "${PYTHON_BIN:-python3}" train.py "$@" ;;
  test) exec "${PYTHON_BIN:-python3}" test.py "$@" ;;
  check) exec "${PYTHON_BIN:-python3}" -m SCOPE "$@" ;;
  *) printf 'Usage: bash run.sh {train|test|check} [arguments]\n' >&2; exit 2 ;;
esac
