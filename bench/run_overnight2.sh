#!/usr/bin/env bash
# Overnight runner v2 — disk-aware, card-safe. Runs the two tracks SEQUENTIALLY
# (both need all 4 cards, so they can't overlap):
#   1. small models (7): cat3 + cat4 + cat5
#   2. 70B models (3): per-model download -> cat2(if needed)+cat3+cat4+cat5 -> delete
# cat2 for the 9 non-dsr1 models is already done. Run in tmux.
set -uo pipefail
REPO="/home/intel/ctao/turboquant/vllm"
echo "================ OVERNIGHT v2 START $(date) ================"
echo ">>> TRACK 1: small models cat3/4/5"
bash "$REPO/bench/run_small_cats345.sh"
echo ">>> TRACK 2: 70B models all categories"
bash "$REPO/bench/run_70b_all_cats.sh"
echo "================ OVERNIGHT v2 COMPLETE $(date) ================"
