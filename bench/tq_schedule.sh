#!/usr/bin/env bash
# Host-side parallel scheduler for the TQ-only perf sweep.
#   Phase 1: TP=1 models — up to 4 in parallel, one per card (0,1,2,3)
#   Phase 2: TP=2 models — up to 2 in parallel, card pairs {0,1} and {2,3}
#   Phase 3: TP=4 models — sequential, all 4 cards; download→run→delete weights
# Each job = `docker exec tq-bench bash tq_job.sh <model> <tp> <card> <port>`,
# which logs internally to tq_perf/driver_logs/<model>_tp<tp>.log.
# Run inside tmux.
set -uo pipefail

NAME="tq-bench"
REPO="/home/intel/ctao/turboquant/vllm"
JOB="$REPO/bench/tq_job.sh"
HFHOME="/home/intel/models/hf-home"
BASE_PORT=8192

# Exactly the v0.22.1rc1 FP8-paper 10-model set (Table 3 / Table 6).
# TP=1 (4): Llama-3.1-8B, DeepSeek-R1-7B, Gemma-3-1B, Qwen3-8B
TP1=(llama31 deepseekr1 gemma3 qwen3)
# TP=2 (3): Gemma-4-E4B, Qwen2.5-14B, Mistral-Small-24B
TP2=(gemma4 qwen25 mistral)
# TP=4 jobs (all four cards), sequential with weight deletion
TP4=(llama33_70b qwen25_72b deepseekr1_70b)
declare -A HF=(
  [llama33_70b]="meta-llama/Llama-3.3-70B-Instruct"
  [qwen25_72b]="Qwen/Qwen2.5-72B-Instruct"
  [deepseekr1_70b]="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)

# Run a batch in parallel: SLOTS jobs at a time, each on consecutive cards.
# args: TP SLOTS model...
run_parallel() {
  local tp="$1" slots="$2"; shift 2
  local models=("$@")
  local i=0
  while (( i < ${#models[@]} )); do
    local pids=()
    for (( s=0; s<slots && i<${#models[@]}; s++, i++ )); do
      local ms="${models[$i]}"
      local card=$(( s * tp ))
      local port=$(( BASE_PORT + s ))
      echo "$(date +%H:%M:%S) [TP$tp] launch $ms card=$card port=$port"
      docker exec "$NAME" bash "$JOB" "$ms" "$tp" "$card" "$port" >/dev/null 2>&1 &
      pids+=( $! )
    done
    # wait for this wave before starting the next
    for p in "${pids[@]}"; do wait "$p" || true; done
    echo "$(date +%H:%M:%S) [TP$tp] wave complete"
  done
}

echo "==== TQ PARALLEL SWEEP START $(date) ===="

echo "---- Phase 1: TP=1, 4-wide ----"
run_parallel 1 4 "${TP1[@]}"

echo "---- Phase 2: TP=2, 2-wide ----"
run_parallel 2 2 "${TP2[@]}"

echo "---- Phase 3: TP=4, sequential (download→run→delete) ----"
for ms in "${TP4[@]}"; do
  echo "$(date +%H:%M:%S) [TP4] $ms"
  docker exec "$NAME" bash "$JOB" "$ms" 4 0 "$BASE_PORT" >/dev/null 2>&1 || true
  dir="models--$(echo "${HF[$ms]}" | sed 's#/#--#g')"
  echo "freeing $HFHOME/hub/$dir"
  docker exec "$NAME" bash -lc "rm -rf '$HFHOME/hub/$dir'" || true
  df -h /home/intel | tail -1
done

echo "==== TQ PARALLEL SWEEP DONE $(date) ===="
