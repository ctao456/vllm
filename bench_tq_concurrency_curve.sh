#!/bin/bash
# bench_tq_concurrency_curve.sh — Argument 2 (Concurrency): TPS vs concurrency.
#
# For each (model, config), sweep concurrency ∈ {1,4,8,16,32,64,128,256} at
# max_model_len=4096. Plot output TPS vs concurrency to show that
# turboquant_4bit_nc sustains higher throughput at concurrency levels
# where BF16 becomes KV-starved.
#
# Per-model TP: Llama-3.1-8B and Gemma-4-E4B use TP=1 (fit on one B70 card),
# Qwen2.5-14B uses TP=2 (needs both cards).
#
# Scheduling: TP=1 models run in PARALLEL on separate GPUs (GPU 0, GPU 1),
# then TP=2 models run sequentially using both GPUs.
#
# Usage:
#   bash bench_tq_concurrency_curve.sh                       # all 3 models
#   bash bench_tq_concurrency_curve.sh qwen25_14b            # one model
#
# Output:
#   /workspace/bench_results/concurrency_curve_<TS>.csv
set -euo pipefail

CONTAINER="vllm-test"
BASE_PORT=8200
MAX_MODEL_LEN=4096
INPUT_LEN=1024
OUTPUT_LEN=512

CONCURRENCIES=(1 4 8 16 32 64 128 256)
CONFIGS=("bf16" "turboquant_4bit_nc")

RESULT_DIR="/workspace/bench_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CSV_OUT="${RESULT_DIR}/concurrency_curve_${TIMESTAMP}.csv"

# ---------------------------------------------------------------------------
resolve_model() {
    case "$1" in
        gemma4_e4b)  echo "google/gemma-4-E4B-it" ;;
        llama31_8b) echo "meta-llama/Llama-3.1-8B-Instruct" ;;
        qwen25_14b) echo "Qwen/Qwen2.5-14B-Instruct" ;;
        *)          echo "UNKNOWN"; return 1 ;;
    esac
}

resolve_tp() {
    case "$1" in
        gemma4_e4b)  echo 1 ;;
        llama31_8b) echo 1 ;;
        qwen25_14b) echo 2 ;;
        *)          echo 2 ;;
    esac
}

resolve_gpu() {
    case "$1" in
        gemma4_e4b)  echo "0" ;;
        llama31_8b) echo "1" ;;
        *)          echo "0,1" ;;
    esac
}

resolve_port() {
    case "$1" in
        gemma4_e4b)  echo $((BASE_PORT)) ;;
        llama31_8b) echo $((BASE_PORT + 1)) ;;
        qwen25_14b) echo $((BASE_PORT + 2)) ;;
        *)          echo $((BASE_PORT)) ;;
    esac
}

# ---------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
    set -- gemma4_e4b llama31_8b qwen25_14b
fi
MODEL_SHORTS=("$@")
for ms in "${MODEL_SHORTS[@]}"; do
    if ! resolve_model "${ms}" > /dev/null 2>&1; then
        echo "Unknown model: ${ms}. Choose from: gemma4_e4b, llama31_8b, qwen25_14b"; exit 1
    fi
done

# ---------------------------------------------------------------------------
start_server() {
    local model_short="$1" config="$2" model="$3" tp_size="$4" gpu="$5" port="$6"
    local server_log="/tmp/vllm_curve_server_${model_short}.log"

    echo "[${model_short}] Starting: config=${config} TP=${tp_size} GPU=${gpu} port=${port}"

    local kv_arg=""
    [[ "${config}" != "bf16" ]] && kv_arg="--kv-cache-dtype ${config}"

    local serve_cmd="ZE_AFFINITY_MASK=${gpu} \
        VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 \
        vllm serve ${model} \
        --port ${port} \
        --tensor-parallel-size ${tp_size} \
        --dtype bfloat16 \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization 0.92 \
        --enforce-eager \
        --max-num-batched-tokens 8192 \
        --block-size 64 \
        --no-enable-log-requests \
        --no-enable-prefix-caching \
        ${kv_arg} \
        > ${server_log} 2>&1"

    docker exec -d ${CONTAINER} bash -c "${serve_cmd}"

    local attempts=0
    while ! docker exec ${CONTAINER} curl -sf http://localhost:${port}/health > /dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [[ ${attempts} -gt 600 ]]; then
            echo "[${model_short}]   ERROR: server failed to start"
            docker exec ${CONTAINER} tail -60 "${server_log}" 2>/dev/null || true
            stop_server "${model_short}" "${port}"
            return 1
        fi
        sleep 1
    done
    echo "[${model_short}]   Server ready in ${attempts}s"

    docker exec ${CONTAINER} vllm bench serve \
        --model "${model}" --backend openai --endpoint /v1/completions \
        --port "${port}" --dataset-name random \
        --random-input-len 64 --random-output-len 32 \
        --num-prompts 4 --max-concurrency 4 \
        --ignore-eos --disable-tqdm > /dev/null 2>&1 || true
}

stop_server() {
    local model_short="$1" port="$2"
    echo "[${model_short}]   Stopping server on port ${port}..."
    docker exec ${CONTAINER} bash -c "
        pkill -TERM -f 'vllm serve.*--port ${port}' 2>/dev/null || true
        sleep 3
        pkill -KILL -f 'vllm serve.*--port ${port}' 2>/dev/null || true
        sleep 2
        # Only nuke shared resources if no other vllm serve is running
        if ! pgrep -f 'vllm serve' > /dev/null 2>&1; then
            pkill -KILL -f 'EngineCore' 2>/dev/null || true
            pkill -KILL -f 'from multiprocessing.spawn' 2>/dev/null || true
            rm -f /dev/shm/psm_* /dev/shm/sem.loky-* 2>/dev/null || true
        fi
        for i in \$(seq 1 30); do
            ss -tln 2>/dev/null | grep -q ':${port} ' || break
            sleep 1
        done
    " || true
    sleep 5
}

# ---------------------------------------------------------------------------
run_point() {
    local model_short="$1" config="$2" conc="$3" model="$4" port="$5"
    local n_prompts=$(( conc * 4 ))
    [[ ${n_prompts} -lt 100 ]] && n_prompts=100
    [[ ${n_prompts} -gt 1024 ]] && n_prompts=1024

    local tag="curve_${model_short}_${config}_c${conc}_${TIMESTAMP}"
    echo "[${model_short}]   conc=${conc} n=${n_prompts}"

    docker exec ${CONTAINER} vllm bench serve \
        --model "${model}" --backend openai --endpoint /v1/completions \
        --port "${port}" --dataset-name random \
        --random-input-len "${INPUT_LEN}" --random-output-len "${OUTPUT_LEN}" \
        --num-prompts "${n_prompts}" \
        --max-concurrency "${conc}" \
        --ignore-eos --disable-tqdm \
        --save-result --result-dir "${RESULT_DIR}" \
        --result-filename "${tag}.json" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --metadata model="${model}" config="${config}" concurrency="${conc}" \
        > /tmp/bench_${tag}.log 2>&1 || echo "[${model_short}]     FAILED"
}

# ---------------------------------------------------------------------------
run_model_suite() {
    local model_short="$1"
    local model tp_size gpu port
    model=$(resolve_model "${model_short}")
    tp_size=$(resolve_tp "${model_short}")
    gpu=$(resolve_gpu "${model_short}")
    port=$(resolve_port "${model_short}")

    echo ""
    echo "[${model_short}] ============================================"
    echo "[${model_short}]   Model: ${model}  TP=${tp_size}  GPU=${gpu}  port=${port}"
    echo "[${model_short}] ============================================"

    for config in "${CONFIGS[@]}"; do
        echo ""
        echo "[${model_short}] --- Config: ${config} ---"
        if start_server "${model_short}" "${config}" "${model}" "${tp_size}" "${gpu}" "${port}"; then
            for conc in "${CONCURRENCIES[@]}"; do
                run_point "${model_short}" "${config}" "${conc}" "${model}" "${port}" || true
            done
        fi
        stop_server "${model_short}" "${port}"
    done
}

# ---------------------------------------------------------------------------
emit_csv() {
    docker exec ${CONTAINER} mkdir -p "${RESULT_DIR}"
    docker exec ${CONTAINER} bash -c "echo 'model,config,concurrency,num_prompts,request_throughput,output_throughput,mean_ttft_ms,mean_tpot_ms,p90_ttft_ms,p90_tpot_ms,p99_ttft_ms' > ${CSV_OUT}"

    for ms in "${MODEL_SHORTS[@]}"; do
        for config in "${CONFIGS[@]}"; do
            for conc in "${CONCURRENCIES[@]}"; do
                local tag="curve_${ms}_${config}_c${conc}_${TIMESTAMP}"
                docker exec -i ${CONTAINER} python3 - <<PY > /dev/null 2>&1 || true
import json, os
p = "${RESULT_DIR}/${tag}.json"
out = "${CSV_OUT}"
if os.path.exists(p):
    with open(p) as f:
        d = json.load(f)
    row = ",".join(str(x) for x in [
        "${ms}", "${config}", ${conc}, d.get("num_prompts",""),
        f"{d.get('request_throughput',0):.4f}",
        f"{d.get('output_throughput',0):.2f}",
        f"{d.get('mean_ttft_ms',0):.2f}",
        f"{d.get('mean_tpot_ms',0):.2f}",
        f"{d.get('p90_ttft_ms',0):.2f}",
        f"{d.get('p90_tpot_ms',0):.2f}",
        f"{d.get('p99_ttft_ms',0):.2f}",
    ])
    with open(out, "a") as f:
        f.write(row + "\n")
PY
            done
        done
    done

    echo ""
    echo "CSV: ${CSV_OUT}"
    docker exec ${CONTAINER} cat "${CSV_OUT}"
}

# ===========================================================================
echo "============================================"
echo "  TPS-vs-Concurrency Curve"
echo "  Models:   ${MODEL_SHORTS[*]}"
echo "  Configs:  ${CONFIGS[*]}"
echo "  Conc:     ${CONCURRENCIES[*]}"
echo "  ISL/OSL:  ${INPUT_LEN}/${OUTPUT_LEN}  max_model_len=${MAX_MODEL_LEN}"
echo "  Timestamp: ${TIMESTAMP}"
echo "  TP=1 models run in parallel on separate GPUs"
echo "============================================"

docker exec ${CONTAINER} mkdir -p ${RESULT_DIR}

# Split models by TP
TP1_MODELS=()
TP2_MODELS=()
for ms in "${MODEL_SHORTS[@]}"; do
    tp=$(resolve_tp "${ms}")
    if [[ "${tp}" == "1" ]]; then
        TP1_MODELS+=("${ms}")
    else
        TP2_MODELS+=("${ms}")
    fi
done

FAIL=0

# --- Phase 1: TP=1 models in parallel on separate GPUs ---
if [[ ${#TP1_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo ">>> Phase 1: TP=1 models in parallel (${TP1_MODELS[*]})"
    PIDS=()
    for ms in "${TP1_MODELS[@]}"; do
        run_model_suite "${ms}" &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do
        wait "${pid}" || FAIL=1
    done
    echo ""
    echo ">>> Phase 1 complete"
fi

# --- Phase 2: TP=2 models sequentially (need both GPUs) ---
if [[ ${#TP2_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo ">>> Phase 2: TP=2 models sequentially (${TP2_MODELS[*]})"
    docker exec ${CONTAINER} bash -c "
        rm -f /dev/shm/psm_* /dev/shm/sem.loky-* 2>/dev/null || true
    " || true
    sleep 10
    for ms in "${TP2_MODELS[@]}"; do
        run_model_suite "${ms}" || FAIL=1
    done
    echo ""
    echo ">>> Phase 2 complete"
fi

emit_csv
exit ${FAIL}
