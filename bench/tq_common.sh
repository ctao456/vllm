#!/usr/bin/env bash
# Shared library for the TurboQuant 5-category benchmark suite.
# Sourced by cat2_throughput.sh / cat3_sla.sh / cat4_longctx.sh / cat5_ruler.sh,
# all of which run INSIDE the container. Encodes every fix learned from the
# initial sweep (see TQ_VS_FP8_BENCHMARK.md §4):
#   - HF_HUB_DISABLE_XET=1 (Xet CAS 401 on this host/proxy)
#   - setsid + process-group kill; EngineCore reaped from inside the container
#   - gemma-4 --quantization fp8 -> bf16 auto-fallback (XPU MoE GEMM unsupported)
#   - internal logging (root container vs uid-1000 host dir clash)
# TQ-only: CONFIG is always turboquant_4bit_nc.

set -uo pipefail

CONFIG="turboquant_4bit_nc"
RESULT_ROOT="/home/intel/models/bench-results"

export VLLM_MLA_DISABLE=1 VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export HF_HUB_DISABLE_XET=1 HF_HOME=/home/intel/models/hf-home

# --- model registry (all 10 from the v0.22.1rc1 paper) ---------------------
resolve_model() {
  case "$1" in
    llama31)        echo "meta-llama/Llama-3.1-8B-Instruct" ;;
    deepseekr1)     echo "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" ;;
    gemma3)         echo "google/gemma-3-1b-it" ;;
    qwen3)          echo "Qwen/Qwen3-8B" ;;
    gemma4)         echo "google/gemma-4-E4B-it" ;;
    qwen25)         echo "Qwen/Qwen2.5-14B-Instruct" ;;
    mistral)        echo "mistralai/Mistral-Small-24B-Instruct-2501" ;;
    llama33_70b)    echo "meta-llama/Llama-3.3-70B-Instruct" ;;
    qwen25_72b)     echo "Qwen/Qwen2.5-72B-Instruct" ;;
    deepseekr1_70b) echo "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" ;;
    *) echo "UNKNOWN"; return 1 ;;
  esac
}

resolve_tp() {
  case "$1" in
    llama31|deepseekr1|gemma3|qwen3) echo 1 ;;
    gemma4|qwen25|mistral)           echo 2 ;;
    llama33_70b|qwen25_72b|deepseekr1_70b) echo 4 ;;
    *) echo 1 ;;
  esac
}

# Per-model weight-quant / backend flags. fp8 weights everywhere (matches paper
# capacity footprint, lets 70B fit on 4x32GB); gemma4 needs Triton backend.
resolve_extra_args() {
  case "$1" in
    deepseekr1|deepseekr1_70b) echo "--trust-remote-code --quantization fp8" ;;
    gemma4) echo "--trust-remote-code --attention-backend TRITON_ATTN --quantization fp8" ;;
    *) echo "--quantization fp8" ;;
  esac
}

is_70b() { case "$1" in llama33_70b|qwen25_72b|deepseekr1_70b) return 0 ;; *) return 1 ;; esac; }

# ze affinity mask for a model: TP consecutive cards from $card; 70B auto-detect.
ze_mask() {  # model short, base card, tp
  local ms="$1" card="$2" tp="$3"
  if is_70b "$ms"; then echo ""; return; fi
  local m="$card"; for ((k=1;k<tp;k++)); do m="${m},$((card+k))"; done
  echo "ZE_AFFINITY_MASK=${m}"
}

# --- server lifecycle ------------------------------------------------------
# Globals set by tq_start_server: SERVER_PGID, WEIGHTS_USED
SERVER_PGID=""; WEIGHTS_USED="fp8"

tq_cleanup() {  # $1 port
  local port="$1"
  [[ -n "${SERVER_PGID:-}" && "$SERVER_PGID" != "0" ]] && kill -9 -- "-${SERVER_PGID}" 2>/dev/null || true
  pkill -9 -f "vllm serve.*--port ${port}" 2>/dev/null || true
  sleep 4
}

# tq_try_serve <model> <port> <tp> <mask> <maxlen> <gpumem> <maxbatched> <maxseqs> <extra> <logfile>
# Launches one server, waits for /health. Returns 0 healthy, 1 not.
tq_try_serve() {
  local model="$1" port="$2" tp="$3" mask="$4" maxlen="$5" gpumem="$6" maxbatched="$7" maxseqs="$8" extra="$9" log="${10}"
  pkill -9 -f "vllm serve.*--port ${port}" 2>/dev/null; sleep 2
  local seqflag=""; [[ -n "$maxseqs" ]] && seqflag="--max-num-seqs ${maxseqs}"
  local batchflag=""; [[ -n "$maxbatched" ]] && batchflag="--max-num-batched-tokens ${maxbatched}"
  local cmd="${mask} VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 VLLM_MLA_DISABLE=1 VLLM_USE_V1=1 HF_HUB_DISABLE_XET=1 \
    vllm serve ${model} --port ${port} --dtype bfloat16 --tensor-parallel-size ${tp} \
    --max-model-len ${maxlen} --gpu-memory-utilization ${gpumem} --enforce-eager \
    ${batchflag} ${seqflag} --block-size 64 --no-enable-log-requests --no-enable-prefix-caching \
    --kv-cache-dtype ${CONFIG} ${extra} > ${log} 2>&1"
  setsid bash -c "${cmd}" &
  local pid=$!
  SERVER_PGID="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
  local timeout=2400; [[ "$tp" -ge 4 ]] && timeout=7200
  # Stall guard: weights are cached, so a healthy server logs progress steadily.
  # If the log file stops growing for STALL seconds (e.g. a oneCCL collective
  # hang at TP=4 startup, workers spinning at 100% CPU), treat as failed so the
  # caller can retry instead of blocking for the full timeout.
  local stall=600; local last_mtime=0 stuck=0
  for ((i=0;i<timeout;i++)); do
    curl -sf "http://localhost:${port}/health" >/dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    grep -qE "Engine core initialization failed|Unsupported data type for fp8 matmul|raise RuntimeError" "$log" 2>/dev/null && return 1
    if (( i % 30 == 0 )); then
      local m; m=$(stat -c %Y "$log" 2>/dev/null || echo 0)
      if [[ "$m" == "$last_mtime" ]]; then stuck=$((stuck+30)); else stuck=0; last_mtime="$m"; fi
      (( stuck >= stall )) && { echo "  [stall] no log progress ${stall}s — treating as failed"; return 1; }
    fi
    sleep 1
  done
  return 1
}

# tq_start_server <model_short> <port> <maxlen> <gpumem> <maxbatched> <maxseqs> <logfile>
# Resolves model/tp/extra, starts server with the gemma-4 fp8->bf16 fallback.
# Sets WEIGHTS_USED. Returns 0 healthy, 1 fail.
tq_start_server() {
  local ms="$1" port="$2" maxlen="$3" gpumem="$4" maxbatched="$5" maxseqs="$6" log="$7"
  local model tp extra mask
  model="$(resolve_model "$ms")"; tp="$(resolve_tp "$ms")"
  extra="$(resolve_extra_args "$ms")"; mask="$(ze_mask "$ms" "${CARD:-0}" "$tp")"
  WEIGHTS_USED="fp8"
  # Try fp8 weights, with one retry to absorb a transient TP=4 oneCCL startup
  # hang (caught by the stall guard). Skip retry if it's the fp8-GEMM error.
  local attempt
  for attempt in 1 2; do
    if tq_try_serve "$model" "$port" "$tp" "$mask" "$maxlen" "$gpumem" "$maxbatched" "$maxseqs" "$extra" "$log"; then
      return 0
    fi
    grep -q "Unsupported data type for fp8 matmul" "$log" 2>/dev/null && break
    [[ "$attempt" == 1 ]] && { echo "  [${ms}] start failed/stalled — cleanup + retry"; tq_cleanup "$port"; }
  done
  if grep -q "Unsupported data type for fp8 matmul" "$log" 2>/dev/null && [[ "$extra" == *"--quantization fp8"* ]]; then
    cp "$log" "${log%.log}.fp8fail.log" 2>/dev/null || true
    local efb; efb="$(echo "$extra" | sed 's/--quantization fp8//')"
    echo "  [${ms}] fp8 weights unsupported -> retrying bf16 weights"
    tq_cleanup "$port"
    WEIGHTS_USED="bf16_fallback"
    tq_try_serve "$model" "$port" "$tp" "$mask" "$maxlen" "$gpumem" "$maxbatched" "$maxseqs" "$efb" "$log" && return 0
  fi
  return 1
}

tq_capacity() {  # logfile -> echoes "tokens conc"
  local kv mc
  kv=$(grep -oP 'GPU KV cache size: \K[0-9,]+' "$1" 2>/dev/null | tr -d ',' | tail -1)
  mc=$(grep -oP 'Maximum concurrency for [0-9,]+ tokens per request: \K[0-9.]+' "$1" 2>/dev/null | tail -1)
  echo "${kv:-NA} ${mc:-NA}"
}

tq_warmup() {  # model port
  vllm bench serve --model "$1" --backend openai --endpoint /v1/completions --port "$2" \
    --dataset-name random --random-input-len 64 --random-output-len 32 \
    --num-prompts 4 --max-concurrency 4 --ignore-eos --disable-tqdm >/dev/null 2>&1 || true
}

# tq_bench <model> <port> <isl> <osl> <nprompts> <conc> <result_dir> <tag>
tq_bench() {
  vllm bench serve --model "$1" --backend openai --endpoint /v1/completions --port "$2" \
    --dataset-name random --random-input-len "$3" --random-output-len "$4" \
    --num-prompts "$5" --max-concurrency "$6" --ignore-eos --disable-tqdm \
    --save-result --result-dir "$7" --result-filename "$8.json" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --metadata model="$1" config="$CONFIG" "${@:9}"
}
