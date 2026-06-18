#!/usr/bin/env bash
# Run ALL remaining categories for the three 70B models, one model at a time,
# deleting each model's weights before the next (disk can't hold >1 70B + smalls).
# Per model: download (if missing) -> cat2(remaining 6-scenario, only if no jsons)
# -> cat3 SLA -> cat4 long-context -> cat5 RULER -> delete weights.
# Run in tmux (separate from the small-model track; they don't share cards only
# if run at different times — see note). 70B use all 4 cards (TP=4).
set -uo pipefail
NAME="tq-bench"; REPO="/home/intel/ctao/turboquant/vllm"
HFHUB="/home/intel/models/hf-home/hub"
HF_TOKEN="$(python3 -c "import json;print(json.load(open('/home/intel/ctao/hf_token.json'))['hf_token'])")"

declare -A REPO_DIR=(
  [llama33_70b]="models--meta-llama--Llama-3.3-70B-Instruct"
  [qwen25_72b]="models--Qwen--Qwen2.5-72B-Instruct"
  [deepseekr1_70b]="models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B")
declare -A HF_ID=(
  [llama33_70b]="meta-llama/Llama-3.3-70B-Instruct"
  [qwen25_72b]="Qwen/Qwen2.5-72B-Instruct"
  [deepseekr1_70b]="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

ORDER=(deepseekr1_70b llama33_70b qwen25_72b)   # dsr1 already downloaded → first

ensure_downloaded() {
  local ms="$1" dir="$HFHUB/${REPO_DIR[$ms]}"
  if [[ -d "$dir" ]] && ! find "$dir" -name '*.incomplete' 2>/dev/null | grep -q . \
     && ls "$dir"/snapshots/*/*.safetensors >/dev/null 2>&1; then
    echo "$(date +%H:%M:%S) $ms already downloaded"; return 0
  fi
  echo "$(date +%H:%M:%S) downloading ${HF_ID[$ms]} ..."
  docker exec tq-bench bash -lc "export HF_HUB_DISABLE_XET=1 HF_HOME=/home/intel/models/hf-home HF_TOKEN='$HF_TOKEN'
    hf download '${HF_ID[$ms]}'" >/dev/null 2>&1
  echo "$(date +%H:%M:%S) $ms download done ($(du -sh "$dir" 2>/dev/null|cut -f1))"
}

for ms in "${ORDER[@]}"; do
  echo "################ $(date) 70B $ms — all categories ################"
  ensure_downloaded "$ms"
  df -h /home/intel | tail -1
  # cat2 only if not already done in the earlier sweep
  if ! ls /home/intel/models/bench-results/cat2_throughput/${ms}_*.json >/dev/null 2>&1; then
    echo "$(date +%H:%M:%S) [$ms] cat2"; docker exec "$NAME" bash "$REPO/bench/cat2_throughput.sh" "$ms" 0 >/dev/null 2>&1 || true
  fi
  echo "$(date +%H:%M:%S) [$ms] cat3"; docker exec "$NAME" bash "$REPO/bench/cat3_sla.sh"      "$ms" 0 >/dev/null 2>&1 || true
  echo "$(date +%H:%M:%S) [$ms] cat4"; docker exec "$NAME" bash "$REPO/bench/cat4_longctx.sh"  "$ms" 0 >/dev/null 2>&1 || true
  echo "$(date +%H:%M:%S) [$ms] cat5"; docker exec "$NAME" bash "$REPO/bench/cat5_ruler.sh"    "$ms" 0 >/dev/null 2>&1 || true
  # Keep weights by default (disk has room). Only delete if free space is tight
  # (< 200 GB) so the next 70B download can't fill the disk.
  free_gb=$(df -BG --output=avail /home/intel | tail -1 | tr -dc '0-9')
  if (( free_gb < 200 )); then
    echo "$(date +%H:%M:%S) [$ms] free=${free_gb}G < 200G -> deleting weights"
    docker exec "$NAME" bash -lc "rm -rf '$HFHUB/${REPO_DIR[$ms]}'" || true
  else
    echo "$(date +%H:%M:%S) [$ms] free=${free_gb}G -> keeping weights"
  fi
  df -h /home/intel | tail -1
done
echo "================ 70B all-categories COMPLETE $(date) ================"
