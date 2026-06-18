#!/usr/bin/env bash
# Host-side master orchestrator for the overnight TQ benchmark (categories 2-5).
# Schedules each category across all 10 models with card-aware parallelism:
#   TP=1 models  -> 4-wide (cards 0,1,2,3)
#   TP=2 models  -> 2-wide (card pairs {0,1},{2,3})
#   TP=4 models  -> sequential (all 4 cards)
# Each job = `docker exec tq-bench bash bench/cat<N>_*.sh <model> <card>` (jobs log
# internally). Categories run in order 2 -> 3 -> 4 -> 5. Run in tmux.
#
# Usage:  bash run_overnight.sh [cat2] [cat3] [cat4] [cat5]   (default: all)
set -uo pipefail
NAME="tq-bench"
REPO="/home/intel/ctao/turboquant/vllm"

TP1=(llama31 deepseekr1 gemma3 qwen3)
TP2=(gemma4 qwen25 mistral)
TP4=(llama33_70b qwen25_72b deepseekr1_70b)

CATS=("$@"); [[ ${#CATS[@]} -eq 0 ]] && CATS=(cat2 cat3 cat4 cat5)

script_for() { case "$1" in
  cat2) echo "bench/cat2_throughput.sh" ;; cat3) echo "bench/cat3_sla.sh" ;;
  cat4) echo "bench/cat4_longctx.sh" ;; cat5) echo "bench/cat5_ruler.sh" ;; esac; }

HFHUB="/home/intel/models/hf-home/hub"
declare -A HFREPO=(
  [llama33_70b]="models--meta-llama--Llama-3.3-70B-Instruct"
  [qwen25_72b]="models--Qwen--Qwen2.5-72B-Instruct"
  [deepseekr1_70b]="models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B")

# Block until a model's weights are fully present (no *.incomplete) so a TP=4 job
# never races the background downloader on the same cache dir.
wait_model_ready() {
  local ms="$1" dir="$HFHUB/${HFREPO[$ms]}"
  echo "$(date +%H:%M:%S) waiting for $ms weights to finish downloading..."
  while :; do
    if [[ -d "$dir" ]] \
       && ! find "$dir" -name '*.incomplete' 2>/dev/null | grep -q . \
       && find "$dir/snapshots" -name 'model*.safetensors' 2>/dev/null | grep -q .; then
      # stable size check: same size twice 30s apart => download settled
      local s1 s2; s1=$(du -s "$dir" 2>/dev/null | cut -f1); sleep 30
      s2=$(du -s "$dir" 2>/dev/null | cut -f1)
      [[ "$s1" == "$s2" ]] && { echo "$(date +%H:%M:%S) $ms ready ($(du -sh "$dir"|cut -f1))"; return 0; }
    fi
    sleep 30
  done
}

# run a list of models, N at a time, each on card = slot*tp
run_wave() {  # script tp slots model...
  local script="$1" tp="$2" slots="$3"; shift 3
  local models=("$@") i=0
  while (( i < ${#models[@]} )); do
    local pids=()
    for (( s=0; s<slots && i<${#models[@]}; s++, i++ )); do
      local ms="${models[$i]}" card=$(( s * tp ))
      echo "$(date +%H:%M:%S) launch $ms (tp$tp card$card) [$script]"
      docker exec "$NAME" bash "$REPO/$script" "$ms" "$card" >/dev/null 2>&1 &
      pids+=( $! )
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    echo "$(date +%H:%M:%S) wave complete"
  done
}

for cat in "${CATS[@]}"; do
  script="$(script_for "$cat")"
  echo "################ $(date) START $cat ($script) ################"
  # cat5 (RULER) is CPU-heavy + long; the author runs it strictly sequential.
  if [[ "$cat" == "cat5" ]]; then
    for ms in "${TP1[@]}" "${TP2[@]}"; do
      echo "$(date +%H:%M:%S) [seq] $ms"
      docker exec "$NAME" bash "$REPO/$script" "$ms" 0 >/dev/null 2>&1 || true
    done
    for ms in "${TP4[@]}"; do
      wait_model_ready "$ms"
      echo "$(date +%H:%M:%S) [seq tp4] $ms"
      docker exec "$NAME" bash "$REPO/$script" "$ms" 0 >/dev/null 2>&1 || true
    done
  else
    run_wave "$script" 1 4 "${TP1[@]}"
    run_wave "$script" 2 2 "${TP2[@]}"
    for ms in "${TP4[@]}"; do
      wait_model_ready "$ms"
      echo "$(date +%H:%M:%S) [tp4] $ms"
      docker exec "$NAME" bash "$REPO/$script" "$ms" 0 >/dev/null 2>&1 || true
    done
  fi
  echo "################ $(date) DONE $cat ################"
done
echo "================ OVERNIGHT SWEEP COMPLETE $(date) ================"
