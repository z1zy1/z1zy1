#!/usr/bin/env bash
set -euo pipefail
mkdir -p experiments/second_cc_mmodalcc_comparison
if [ ! -f external_results/second_cc_mmodalcc_results.csv ]; then
  echo "MModalCC comparison is external only. Provide external_results/second_cc_mmodalcc_results.csv; no result will be fabricated." >&2
  exit 0
fi
cp external_results/second_cc_mmodalcc_results.csv experiments/second_cc_mmodalcc_comparison/external_results.csv
echo "Recorded external MModalCC results at experiments/second_cc_mmodalcc_comparison/external_results.csv"
