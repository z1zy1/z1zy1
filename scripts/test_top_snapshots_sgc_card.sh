#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

EXP_PATH=""
TAGS=(checkpoint_7000 checkpoint_8000 checkpoint_9000 best_balanced best_balanced_v2)

usage() {
  echo "Usage: bash scripts/test_top_snapshots_sgc_card.sh --exp_dir EXP_DIR [--tags tag1,tag2,...]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --exp_dir)
      EXP_PATH="$2"
      shift 2
      ;;
    --tags)
      IFS=',' read -r -a TAGS <<< "$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [ -z "$EXP_PATH" ]; then
  usage
  exit 2
fi

EXP_PATH="${EXP_PATH%/}"
SUMMARY_CSV="$EXP_PATH/test_top_snapshots_summary.csv"
mkdir -p "$EXP_PATH"

printf 'tag,snapshot_path,Bleu_1,Bleu_2,Bleu_3,Bleu_4,METEOR,ROUGE_L,CIDEr,SPICE,delta_Bleu_4,delta_METEOR,delta_ROUGE_L,delta_CIDEr,delta_SPICE,all_above_test_baseline\n' > "$SUMMARY_CSV"

resolve_snapshot() {
  local tag="$1"
  local checkpoint="$tag"
  case "$tag" in
    best_balanced)
      checkpoint="$EXP_PATH/best_balanced.pth"
      ;;
    best_balanced_v2)
      checkpoint="$EXP_PATH/best_balanced_v2.pth"
      ;;
  esac

  EXP_PATH="$EXP_PATH" CHECKPOINT="$checkpoint" TAG="$tag" python -c '
import csv
import glob
import os
import re
import sys

exp_path = os.environ["EXP_PATH"]
checkpoint = os.environ["CHECKPOINT"]
tag = os.environ["TAG"]

def num(text):
    m = re.findall(r"(\d+)", text or "")
    return m[-1] if m else None

def first(paths):
    for path in paths:
        if path and os.path.exists(path):
            return os.path.normpath(path)
    return None

paths = [
    checkpoint,
    os.path.abspath(checkpoint),
    os.path.join(exp_path, checkpoint),
    os.path.join(exp_path, os.path.basename(checkpoint)),
    os.path.join(exp_path, "snapshots", os.path.basename(checkpoint)),
    os.path.join(exp_path, "checkpoints", os.path.basename(checkpoint)),
]
root, ext = os.path.splitext(checkpoint)
if ext.lower() == ".pth":
    paths.append(root + ".pt")
elif ext.lower() == ".pt":
    paths.append(root + ".pth")
found = first(paths)
if found:
    print(found)
    sys.exit(0)

number = num(checkpoint) or num(tag)
patterns = []
if number:
    patterns += [
        os.path.join(exp_path, "snapshots", "*" + number + "*.pt"),
        os.path.join(exp_path, "snapshots", "*" + number + "*.pth"),
        os.path.join(exp_path, "checkpoints", "*" + number + "*.pt"),
        os.path.join(exp_path, "checkpoints", "*" + number + "*.pth"),
    ]

csv_path = os.path.join(exp_path, "eval_snapshots.csv")
if os.path.exists(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row.get("snapshot_path", "")
            if (number and number in os.path.basename(raw)) or tag in os.path.basename(raw):
                patterns.extend([
                    raw,
                    os.path.join(exp_path, raw),
                    os.path.join(exp_path, "snapshots", os.path.basename(raw)),
                ])

matches = []
for pattern in patterns:
    matches.extend(glob.glob(pattern))
matches = sorted({os.path.normpath(path) for path in matches if os.path.exists(path)})
if matches:
    print(matches[0])
    sys.exit(0)
sys.exit(1)
'
}

append_summary_row() {
  local tag="$1"
  local result_json="$2"
  SUMMARY_CSV="$SUMMARY_CSV" TAG="$tag" RESULT_JSON="$result_json" python -c '
import csv
import json
import os

summary_csv = os.environ["SUMMARY_CSV"]
tag = os.environ["TAG"]
result_json = os.environ["RESULT_JSON"]

with open(result_json, encoding="utf-8") as f:
    data = json.load(f)

metrics = data.get("metrics", {})
deltas = data.get("deltas", {})
row = [
    tag,
    data.get("snapshot_path", ""),
    metrics.get("Bleu_1", ""),
    metrics.get("Bleu_2", ""),
    metrics.get("Bleu_3", ""),
    metrics.get("Bleu_4", ""),
    metrics.get("METEOR", ""),
    metrics.get("ROUGE_L", ""),
    metrics.get("CIDEr", ""),
    metrics.get("SPICE", ""),
    deltas.get("Bleu_4", ""),
    deltas.get("METEOR", ""),
    deltas.get("ROUGE_L", ""),
    deltas.get("CIDEr", ""),
    deltas.get("SPICE", ""),
    data.get("ALL_TEST_METRICS_ABOVE_BASELINE", ""),
]
with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(row)
'
}

for tag in "${TAGS[@]}"; do
  if snapshot_path="$(resolve_snapshot "$tag")"; then
    echo "========== test $tag =========="
    bash scripts/test_specific_snapshot_sgc_card.sh \
      --exp_dir "$EXP_PATH" \
      --checkpoint "$snapshot_path" \
      --tag "$tag"
    append_summary_row "$tag" "$EXP_PATH/test_${tag}_result.json"
  else
    echo "Warning: snapshot for tag '$tag' was not found under $EXP_PATH; skipped." >&2
  fi
done

echo "Wrote summary: $SUMMARY_CSV"
