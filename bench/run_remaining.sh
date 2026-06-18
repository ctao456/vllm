#!/usr/bin/env bash
# Finish the overnight sweep: cat5 RULER for the 7 small models (sequential),
# then TRACK 2 (the three 70B models, all remaining categories, download-aware).
# wonderwords/nltk now installed so RULER works. Run in tmux.
set -uo pipefail
NAME="tq-bench"; REPO="/home/intel/ctao/turboquant/vllm"
SMALL=(llama31 deepseekr1 gemma3 qwen3 gemma4 qwen25 mistral)

echo "================ RUN REMAINING START $(date) ================"
echo ">>> cat5 RULER (small models, sequential)"
for ms in "${SMALL[@]}"; do
  echo "$(date +%H:%M:%S) [cat5] $ms"
  docker exec "$NAME" bash "$REPO/bench/cat5_ruler.sh" "$ms" 0 >/dev/null 2>&1 || true
done
echo ">>> TRACK 2: 70B models, all categories"
bash "$REPO/bench/run_70b_all_cats.sh"
echo "================ RUN REMAINING COMPLETE $(date) ================"
