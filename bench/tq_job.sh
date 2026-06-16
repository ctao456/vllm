#!/usr/bin/env bash
# Single TQ benchmark job: one (model, TP) pinned to a base card + port.
# Runs INSIDE the container. Mirrors the FP8 author's harness exactly
# (same serve flags, same 6 scenarios, --quantization fp8 weights) but with
# --kv-cache-dtype turboquant_4bit_nc instead of fp8.  TQ-only (no bf16/fp8 KV).
#
# Args: MODEL_SHORT TP CARD PORT
set -uo pipefail

MODEL_SHORT="$1"; TP="$2"; CARD="$3"; PORT="$4"
CONFIG="turboquant_4bit_nc"
MAX_MODEL_LEN=4096          # v0.22.1 paper Category-1 capacity config
RESULT_DIR="/home/intel/models/bench-results/tq_perf"
LOG_DIR="$RESULT_DIR/logs"
DRV_DIR="$RESULT_DIR/driver_logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR" "$LOG_DIR" "$DRV_DIR"

# All stdout/stderr → this job's own driver log (root-owned dir is fine; root
# can write). Host scheduler does NOT redirect, avoiding host/container uid clash.
exec > >(tee -a "$DRV_DIR/${MODEL_SHORT}_tp${TP}.log") 2>&1

export VLLM_MLA_DISABLE=1 VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
# Xet CAS backend returns 401 on this host/proxy; use classic LFS download path.
export HF_HUB_DISABLE_XET=1

# 6 scenarios (FP8/TQ author suite): name:ISL:OSL:N:concurrency
SCENARIOS=(
  "short_decode:128:512:200:32"
  "long_prefill:4096:128:200:32"
  "mixed:512:512:200:32"
  "high_load:512:128:500:64"
  "very_long_prefill:7168:64:200:16"
  "decode_heavy:64:1024:200:32"
)

resolve_model() {
  case "$1" in
    llama31)        echo "meta-llama/Llama-3.1-8B-Instruct" ;;
    llama31_fp8)    echo "nvidia/Llama-3.1-8B-Instruct-FP8" ;;
    deepseekr1)     echo "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" ;;
    gemma3)         echo "google/gemma-3-1b-it" ;;
    gemma4)         echo "google/gemma-4-E4B-it" ;;
    qwen3)          echo "Qwen/Qwen3-8B" ;;
    qwen25)         echo "Qwen/Qwen2.5-14B-Instruct" ;;
    mistral)        echo "mistralai/Mistral-Small-24B-Instruct-2501" ;;
    llama33_70b)    echo "meta-llama/Llama-3.3-70B-Instruct" ;;
    qwen25_72b)     echo "Qwen/Qwen2.5-72B-Instruct" ;;
    deepseekr1_70b) echo "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" ;;
    *) echo "UNKNOWN"; return 1 ;;
  esac
}

# Per-model extra args, matching the FP8 author (online --quantization fp8 weight
# quant for every model; deepseek needs trust-remote-code; gemma4 needs the Triton
# backend for heterogeneous head dims). FP8 weights also let the 70B models fit on
# 4x32GB. Gemma-4 + fp8 is unsupported by the XPU fp8 GEMM kernel, so the run loop
# auto-falls-back to bf16 weights for it (flagged via WEIGHTS=bf16_fallback).
resolve_extra_args() {
  case "$1" in
    deepseekr1|deepseekr1_70b) echo "--trust-remote-code --quantization fp8" ;;
    gemma4) echo "--trust-remote-code --attention-backend TRITON_ATTN --quantization fp8" ;;
    *) echo "--quantization fp8" ;;
  esac
}

MODEL="$(resolve_model "$MODEL_SHORT")"
EXTRA="$(resolve_extra_args "$MODEL_SHORT")"
TAGBASE="${MODEL_SHORT}_${CONFIG}_tp${TP}"
SERVER_LOG="$LOG_DIR/server_${TAGBASE}.log"

# Affinity mask: TP consecutive cards from CARD. 70B models auto-detect (no mask).
ZE_PREFIX=""
if [[ "$MODEL_SHORT" != "llama33_70b" && "$MODEL_SHORT" != "qwen25_72b" && "$MODEL_SHORT" != "deepseekr1_70b" ]]; then
  mask="$CARD"
  for ((k=1;k<TP;k++)); do mask="${mask},$((CARD+k))"; done
  ZE_PREFIX="ZE_AFFINITY_MASK=${mask}"
fi
# --dtype bfloat16 sets the compute dtype; weight quantization (fp8) is applied
# separately via EXTRA (resolve_extra_args). KV cache is TurboQuant (CONFIG).
DTYPE="--dtype bfloat16"

SERVER_PGID=""
cleanup() {
  if [[ -n "${SERVER_PGID:-}" && "$SERVER_PGID" != "0" ]]; then
    kill -9 -- "-${SERVER_PGID}" 2>/dev/null || true
  fi
  pkill -9 -f "vllm serve.*--port $PORT" 2>/dev/null || true
  sleep 3
}
trap cleanup EXIT

# start_server <extra_args> : launch + wait for health. Returns 0 healthy, 1 not.
start_server() {
  local extra="$1"
  pkill -9 -f "vllm serve.*--port $PORT" 2>/dev/null; sleep 2
  # v0.22.1 paper serve config: gpu-util 0.90, block-size 64, enforce-eager;
  # no max-num-batched-tokens / max-num-seq overrides (uses vLLM defaults).
  local serve_cmd="${ZE_PREFIX} VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 VLLM_MLA_DISABLE=1 VLLM_USE_V1=1 \
    vllm serve ${MODEL} --port ${PORT} ${DTYPE} --tensor-parallel-size ${TP} \
    --max-model-len ${MAX_MODEL_LEN} --gpu-memory-utilization 0.90 --enforce-eager \
    --block-size 64 --no-enable-log-requests --no-enable-prefix-caching \
    --kv-cache-dtype ${CONFIG} ${extra} > ${SERVER_LOG} 2>&1"
  setsid bash -c "${serve_cmd}" &
  local pid=$!
  SERVER_PGID="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
  local timeout=2400; [[ "$TP" -ge 4 ]] && timeout=7200
  for ((i=0;i<timeout;i++)); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || return 1   # launcher exited → server died
    # Fatal engine errors keep the launcher alive but never recover; detect them
    # in the log so the fallback fires within seconds instead of after `timeout`.
    if grep -qE "Engine core initialization failed|Unsupported data type for fp8 matmul|EngineCore failed to start|raise RuntimeError" "$SERVER_LOG" 2>/dev/null; then
      return 1
    fi
    sleep 1
  done
  return 1
}

echo "[$MODEL_SHORT tp$TP] start card=$CARD port=$PORT mask='${ZE_PREFIX}' extra='${EXTRA}'"
if ! start_server "$EXTRA"; then
  # Auto-fallback ONLY for the genuine XPU fp8-GEMM rejection (Gemma-4 MoE).
  # Must NOT trigger on download/auth/OOM errors (those need a real fix, not bf16).
  if grep -q "Unsupported data type for fp8 matmul" "$SERVER_LOG" 2>/dev/null \
     && [[ "$EXTRA" == *"--quantization fp8"* ]]; then
    EXTRA_FB="$(echo "$EXTRA" | sed 's/--quantization fp8//')"
    # Preserve the fp8 failure log before the bf16 retry overwrites SERVER_LOG.
    cp "$SERVER_LOG" "${SERVER_LOG%.log}.fp8fail.log" 2>/dev/null || true
    echo "[$MODEL_SHORT tp$TP] fp8 weights unsupported; retrying bf16 weights"
    cleanup
    if start_server "$EXTRA_FB"; then
      echo "bf16_fallback" > "$LOG_DIR/weights_${TAGBASE}.txt"
    else
      echo "[$MODEL_SHORT tp$TP] SERVER FAILED (after fallback)"; tail -30 "$SERVER_LOG"
      echo "FAIL" > "$LOG_DIR/kv_${TAGBASE}.txt"; exit 1
    fi
  else
    echo "[$MODEL_SHORT tp$TP] SERVER FAILED"; tail -30 "$SERVER_LOG"
    echo "FAIL" > "$LOG_DIR/kv_${TAGBASE}.txt"; exit 1
  fi
fi

KV=$(grep -oP 'GPU KV cache size: \K[0-9,]+' "$SERVER_LOG" | tr -d ',' | tail -1)
MC=$(grep -oP 'Maximum concurrency for [0-9,]+ tokens per request: \K[0-9.]+' "$SERVER_LOG" | tail -1)
echo "${KV:-NA}" > "$LOG_DIR/kv_${TAGBASE}.txt"
echo "[$MODEL_SHORT tp$TP] KV cache tokens: ${KV:-NA}  max_conc: ${MC:-NA}x"

# warmup
vllm bench serve --model "$MODEL" --backend openai --endpoint /v1/completions --port "$PORT" \
  --dataset-name random --random-input-len 64 --random-output-len 32 \
  --num-prompts 3 --max-concurrency 4 --ignore-eos --disable-tqdm >/dev/null 2>&1 || true

for spec in "${SCENARIOS[@]}"; do
  IFS=':' read -r name isl osl n conc <<< "$spec"
  if (( isl + osl > MAX_MODEL_LEN )); then echo "[$MODEL_SHORT tp$TP] [$name] SKIP (>$MAX_MODEL_LEN)"; continue; fi
  tag="${TAGBASE}__${name}__${TIMESTAMP}"
  echo "[$MODEL_SHORT tp$TP] [$name] in=$isl out=$osl n=$n conc=$conc"
  vllm bench serve --model "$MODEL" --backend openai --endpoint /v1/completions --port "$PORT" \
    --dataset-name random --random-input-len "$isl" --random-output-len "$osl" \
    --num-prompts "$n" --max-concurrency "$conc" --ignore-eos \
    --save-result --result-dir "$RESULT_DIR" --result-filename "${tag}.json" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --metadata model="$MODEL" config="$CONFIG" scenario="$name" tp="$TP" \
    > "$LOG_DIR/scenario_${tag}.log" 2>&1 \
    && echo "[$MODEL_SHORT tp$TP] [$name] done" \
    || echo "[$MODEL_SHORT tp$TP] [$name] FAILED"
done

echo "[$MODEL_SHORT tp$TP] DONE"   # trap cleanup() reaps the process group
