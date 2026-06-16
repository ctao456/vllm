#!/usr/bin/env bash
# Run the two remaining 70B models sequentially (qwen25_72b, deepseekr1_70b),
# deleting each model's HF cache after its run to respect the disk budget.
# Standalone (the main scheduler already finished). Run inside tmux.
set -uo pipefail
NAME="tq-bench"
JOB="/home/intel/ctao/turboquant/vllm/bench/tq_job.sh"
HFHOME="/home/intel/models/hf-home"

declare -A HF=(
  [qwen25_72b]="Qwen/Qwen2.5-72B-Instruct"
  [deepseekr1_70b]="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)

for ms in qwen25_72b deepseekr1_70b; do
  echo "######## $(date +%H:%M:%S) $ms ########"
  docker exec "$NAME" bash "$JOB" "$ms" 4 0 8192 || echo "$ms job exited nonzero"
  dir="models--$(echo "${HF[$ms]}" | sed 's#/#--#g')"
  echo "freeing $HFHOME/hub/$dir"
  docker exec "$NAME" bash -lc "rm -rf '$HFHOME/hub/$dir'" || true
  df -h /home/intel | tail -1
done
echo "######## remaining 70B done $(date +%H:%M:%S) ########"
