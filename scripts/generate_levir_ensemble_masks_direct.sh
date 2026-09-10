#!/usr/bin/env bash
set -euo pipefail

# Single-entry replacement flow. The underlying script deletes the old
# pseudo-mask directories before inference to keep disk usage bounded.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

exec bash "$SCRIPT_DIR/generate_levir_ensemble_masks.sh" \
  --delete-existing "$@"
