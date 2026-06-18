#!/usr/bin/env bash
# Run cat3 (SLA), cat4 (long-context), cat5 (RULER) for the 7 SMALL models only
# (all cached, no disk pressure). TP=1 4-wide, TP=2 2-wide for cat3/cat4; cat5
# sequential (lm_eval is CPU-heavy). Run in tmux. 70B handled separately.
set -uo pipefail
NAME="tq-bench"; REPO="/home/intel/ctao/turboquant/vllm"
TP1=(llama31 deepseekr1 gemma3 qwen3)
TP2=(gemma4 qwen25 mistral)

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

for cat in cat3 cat4 cat5; do
  case "$cat" in
    cat3) script="bench/cat3_sla.sh" ;;
    cat4) script="bench/cat4_longctx.sh" ;;
    cat5) script="bench/cat5_ruler.sh" ;;
  esac
  echo "################ $(date) START $cat (small models) ################"
  if [[ "$cat" == "cat5" ]]; then
    for ms in "${TP1[@]}" "${TP2[@]}"; do
      echo "$(date +%H:%M:%S) [seq] $ms"
      docker exec "$NAME" bash "$REPO/$script" "$ms" 0 >/dev/null 2>&1 || true
    done
  else
    run_wave "$script" 1 4 "${TP1[@]}"
    run_wave "$script" 2 2 "${TP2[@]}"
  fi
  echo "################ $(date) DONE $cat (small models) ################"
done
echo "================ SMALL-MODEL cat3/4/5 COMPLETE $(date) ================"
