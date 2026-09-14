#!/usr/bin/env bash
set -euo pipefail

# Fresh experiment identity for the scheme-1 rerun. Old r1 artifacts remain
# read-only evidence and are never considered by this entry point.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_DIR
export PROTOCOL_ID=p1_rsaca_20260913
export PAIR_ROOT="${PAIR_ROOT:-$PROJECT_DIR/experiments/p1_rsaca_20260914/paired_matrix_r2}"
export P1_REFERENCE_JSON="${P1_REFERENCE_JSON:-$PROJECT_DIR/configs/protocols/p1_rsaca_20260913.json}"
export SEEDS="1111 2222 3333"
export MAX_ITER=10000
export SNAPSHOT_INTERVAL=1000
export SAVE_INTERVAL=1000
export EVAL_INTERVAL=1000

exec bash "$PROJECT_DIR/scripts/run_paired_card_rsaca_matrix.sh" "$@"
