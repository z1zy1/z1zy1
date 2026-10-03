#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential, resumable P1 B/D/C0/G training flow.  A failed stage stops the
# flow and preserves the historical run directory for inspection.
PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python}
EXP_ROOT=${EXP_ROOT:-"$PROJECT_DIR/experiments/p1_semantic_controls_gpu_$(date -u +%Y%m%dT%H%M%SZ)"}
GPU=${PYTORCH_GPU:-0}
NUM_WORKERS=${NUM_WORKERS:-8}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
RUN_LOCKED_TEST=${RUN_LOCKED_TEST:-0}
ONLY_LOCKED_TEST=${ONLY_LOCKED_TEST:-0}
SEMANTIC_SOURCE_LEVIR_CC=${SEMANTIC_SOURCE_LEVIR_CC:-unknown}
SEMANTIC_SOURCE_LEVIR_MCI=${SEMANTIC_SOURCE_LEVIR_MCI:-unknown}
SEMANTIC_SOURCE_SECOND_CC=${SEMANTIC_SOURCE_SECOND_CC:-unknown}
LOG_DIR=${LOG_DIR:-"$PROJECT_DIR/run_logs/semantic_controls_$(date -u +%Y%m%dT%H%M%SZ)"}

case "$SEMANTIC_SOURCE_LEVIR_CC" in annotation|prediction|unknown) ;; *) echo "invalid SEMANTIC_SOURCE_LEVIR_CC" >&2; exit 2 ;; esac
case "$SEMANTIC_SOURCE_LEVIR_MCI" in annotation|prediction|unknown) ;; *) echo "invalid SEMANTIC_SOURCE_LEVIR_MCI" >&2; exit 2 ;; esac
case "$SEMANTIC_SOURCE_SECOND_CC" in annotation|prediction|unknown) ;; *) echo "invalid SEMANTIC_SOURCE_SECOND_CC" >&2; exit 2 ;; esac

if [[ "$EXP_ROOT" != "$PROJECT_DIR/experiments/"* ]]; then
  echo "EXP_ROOT must be an independent directory under $PROJECT_DIR/experiments" >&2
  exit 2
fi
if [[ -e "$EXP_ROOT/frozen.json" ]]; then
  echo "refusing an already frozen experiment: $EXP_ROOT" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
export PROJECT_DIR PYTORCH_GPU="$GPU" NUM_WORKERS OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

run_stage() {
  local name=$1
  shift
  local log="$LOG_DIR/${name}.log"
  echo "[$(date -u +%FT%TZ)] START $name"
  printf '%q ' "$@" | tee "$LOG_DIR/${name}.command.txt"
  printf '\n' | tee -a "$LOG_DIR/${name}.command.txt"
  "$@" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "[$(date -u +%FT%TZ)] FAILED $name status=$status; see $log" >&2
    exit "$status"
  fi
  echo "[$(date -u +%FT%TZ)] DONE $name"
}

COMMON=(--root "$EXP_ROOT")
SOURCES=(--semantic-source "levir_cc=$SEMANTIC_SOURCE_LEVIR_CC"
         --semantic-source "levir_mci=$SEMANTIC_SOURCE_LEVIR_MCI"
         --semantic-source "second_cc=$SEMANTIC_SOURCE_SECOND_CC")

if [[ "$ONLY_LOCKED_TEST" == 1 ]]; then
  if [[ ! -f "$EXP_ROOT/protocol.json" || ! -f "$EXP_ROOT/audit.json" ]]; then
    echo "ONLY_LOCKED_TEST requires an existing protocol.json and audit.json" >&2
    exit 2
  fi
  run_stage freeze "$PYTHON" scripts/run_semantic_controls.py --stage freeze "${COMMON[@]}"
  run_stage test "$PYTHON" scripts/run_semantic_controls.py --stage test "${COMMON[@]}"
  run_stage summary "$PYTHON" scripts/run_semantic_controls.py --stage summary "${COMMON[@]}"
  echo "[$(date -u +%FT%TZ)] COMPLETE LOCKED TEST EXP_ROOT=$EXP_ROOT LOG_DIR=$LOG_DIR"
  exit 0
fi

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  run_stage preflight_dry_run "$PYTHON" scripts/run_semantic_controls.py --stage preflight "${COMMON[@]}" "${SOURCES[@]}" --dry-run
  run_stage train_dry_run "$PYTHON" scripts/run_semantic_controls.py --stage train "${COMMON[@]}" --dry-run
  exit 0
fi

run_stage preflight "$PYTHON" scripts/run_semantic_controls.py --stage preflight "${COMMON[@]}" "${SOURCES[@]}"
run_stage train "$PYTHON" scripts/run_semantic_controls.py --stage train "${COMMON[@]}"
run_stage select "$PYTHON" scripts/run_semantic_controls.py --stage select "${COMMON[@]}"
run_stage audit "$PYTHON" scripts/run_semantic_controls.py --stage audit "${COMMON[@]}"

# This is validation-only and does not load CUDA/checkpoints or alter admission.
run_stage validation "$PYTHON" scripts/semantic_controls_evidence.py validation \
  --experiment-root "$EXP_ROOT" --output-dir "$LOG_DIR/validation_evidence"

if [[ "$RUN_LOCKED_TEST" == 1 ]]; then
  run_stage freeze "$PYTHON" scripts/run_semantic_controls.py --stage freeze "${COMMON[@]}"
  run_stage test "$PYTHON" scripts/run_semantic_controls.py --stage test "${COMMON[@]}"
  run_stage summary "$PYTHON" scripts/run_semantic_controls.py --stage summary "${COMMON[@]}"
else
  echo "[$(date -u +%FT%TZ)] LOCKED TEST SKIPPED; set RUN_LOCKED_TEST=1 only after reviewing audit/validation"
fi

echo "[$(date -u +%FT%TZ)] COMPLETE EXP_ROOT=$EXP_ROOT LOG_DIR=$LOG_DIR"
