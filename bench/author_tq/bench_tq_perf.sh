#!/bin/bash
# bench_tq_perf.sh — Compare bf16 vs all TQ presets across 6 serving scenarios.
#
# Matches the benchmarks/sliding_window suite: same server args, same 6 scenarios
# (short_decode, long_prefill, mixed, high_load, very_long_prefill, decode_heavy),
# same bench parameters (openai backend, /v1/completions, max-concurrency, percentiles).
#
# Supports running multiple models in parallel, each pinned to a separate XPU card.
#
# Usage:
#   bash bench_tq_perf.sh qwen3           # single model on card 0
#   bash bench_tq_perf.sh qwen3 gemma3    # parallel: qwen3→card 0, gemma3→card 1
#   bash bench_tq_perf.sh gemma3 gemma4   # parallel: gemma3→card 0, gemma4→card 1
#
# Runs inside docker container 'vllm-test'.
# Each config: start server → warm up → run 6 scenarios → stop server.
# Reports per-scenario perf tables + KV cache compression ratio.
set -euo pipefail

CONTAINER="vllm-test"
BASE_PORT=8192
MAX_MODEL_LEN=8192

# 6 scenarios from benchmarks/sliding_window/scenarios.tsv
# Format: name:input_len:output_len:num_prompts:concurrency
SCENARIOS=(
    "short_decode:128:512:200:32"
    "long_prefill:4096:128:200:32"
    "mixed:512:512:200:32"
    "high_load:512:128:500:64"
    "very_long_prefill:7168:64:200:16"
    "decode_heavy:64:1024:200:32"
)

# All configs: bf16 baseline + TQ presets (ordered by compression)
ALL_CONFIGS=("bf16" "turboquant_k8v4" "turboquant_4bit_nc" "turboquant_k3v4_nc" "turboquant_3bit_nc")

RESULT_DIR="/workspace/bench_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------
resolve_model() {
    case "$1" in
        qwen3)  echo "Qwen/Qwen3-8B" ;;
        gemma3) echo "google/gemma-3-1b-it" ;;
        gemma4) echo "google/gemma-4-E4B-it" ;;
        *)      echo "UNKNOWN"; return 1 ;;
    esac
}

resolve_extra_args() {
    case "$1" in
        gemma4) echo "--trust-remote-code --attention-backend TRITON_ATTN" ;;
        *)      echo "" ;;
    esac
}

# ---------------------------------------------------------------------------
# Parse model arguments — each gets a card and port
# ---------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
    set -- qwen3
fi
MODEL_SHORTS=("$@")
for ms in "${MODEL_SHORTS[@]}"; do
    if ! resolve_model "${ms}" > /dev/null 2>&1; then
        echo "Unknown model: ${ms}. Choose from: qwen3, gemma3, gemma4"; exit 1
    fi
done

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
start_server() {
    local model_short="$1" config="$2" model="$3" extra_args="$4" card="$5" port="$6"
    local server_log="/tmp/vllm_bench_server_${model_short}.log"

    echo "[${model_short}] Starting server: config=${config} card=${card} port=${port}"

    local kv_arg=""
    if [[ "${config}" != "bf16" ]]; then
        kv_arg="--kv-cache-dtype ${config}"
    fi

    local serve_cmd="ZE_AFFINITY_MASK=${card} VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 \
        vllm serve ${model} \
        --port ${port} \
        --dtype bfloat16 \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization 0.92 \
        --enforce-eager \
        --max-num-batched-tokens 8192 \
        --max-num-seq 64 \
        --block-size 64 \
        --no-enable-log-requests \
        --no-enable-prefix-caching \
        ${kv_arg} \
        ${extra_args} \
        > ${server_log} 2>&1"

    docker exec -d ${CONTAINER} bash -c "${serve_cmd}"

    echo "[${model_short}]     Waiting for server..."
    local attempts=0
    while ! docker exec ${CONTAINER} curl -sf http://localhost:${port}/health > /dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [[ ${attempts} -gt 300 ]]; then
            echo "[${model_short}]     ERROR: Server failed to start after 300s"
            docker exec ${CONTAINER} tail -30 "${server_log}" 2>/dev/null || true
            stop_server "${model_short}" "${port}"
            return 1
        fi
        sleep 1
    done
    echo "[${model_short}]     Server ready (${attempts}s)"

    # Warmup: 3 requests at concurrency 4
    echo "[${model_short}]     Warming up..."
    docker exec ${CONTAINER} vllm bench serve \
        --model "${model}" \
        --backend openai \
        --endpoint /v1/completions \
        --port "${port}" \
        --dataset-name random \
        --random-input-len 64 --random-output-len 32 \
        --num-prompts 3 \
        --max-concurrency 4 \
        --ignore-eos \
        --disable-tqdm \
        > /dev/null 2>&1 || true
}

stop_server() {
    local model_short="$1" port="$2"
    echo "[${model_short}]     Stopping server on port ${port}..."
    docker exec ${CONTAINER} bash -c "pkill -f 'vllm serve.*--port ${port}' 2>/dev/null; sleep 2; pkill -9 -f 'vllm serve.*--port ${port}' 2>/dev/null" || true
    sleep 3
}

# ---------------------------------------------------------------------------
# Run one scenario
# ---------------------------------------------------------------------------
run_scenario() {
    local model_short="$1" config="$2" scenario_spec="$3" model="$4" port="$5"
    IFS=':' read -r name in_len out_len n_prompts conc <<< "${scenario_spec}"
    local tag="${model_short}_${config}__${name}__${TIMESTAMP}"

    local total=$((in_len + out_len))
    if [[ ${total} -gt ${MAX_MODEL_LEN} ]]; then
        echo "[${model_short}]     [${name}] SKIP: in+out=${total} > ${MAX_MODEL_LEN}"
        return 0
    fi

    echo "[${model_short}]     [${name}] in=${in_len} out=${out_len} n=${n_prompts} conc=${conc}"

    docker exec ${CONTAINER} vllm bench serve \
        --model "${model}" \
        --backend openai \
        --endpoint /v1/completions \
        --port "${port}" \
        --dataset-name random \
        --random-input-len "${in_len}" \
        --random-output-len "${out_len}" \
        --num-prompts "${n_prompts}" \
        --max-concurrency "${conc}" \
        --ignore-eos \
        --save-result \
        --result-dir ${RESULT_DIR} \
        --result-filename "${tag}.json" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --metadata \
            model="${model}" \
            config="${config}" \
            scenario="${name}" \
            input_len="${in_len}" \
            output_len="${out_len}" \
            num_prompts="${n_prompts}" \
            concurrency="${conc}" \
        > /tmp/bench_${tag}.log 2>&1
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "[${model_short}]     [${name}] FAILED (exit ${rc})"
    else
        echo "[${model_short}]     [${name}] done"
    fi
}

# ---------------------------------------------------------------------------
# Extract KV cache size from server log
# ---------------------------------------------------------------------------
extract_kv_cache_tokens() {
    local model_short="$1"
    docker exec ${CONTAINER} grep -oP 'GPU KV cache size: \K[0-9,]+' /tmp/vllm_bench_server_${model_short}.log 2>/dev/null \
        | tr -d ',' | tail -1 \
        || echo "N/A"
}

# ---------------------------------------------------------------------------
# run_model_suite — all configs × scenarios for one model (runs in background)
# ---------------------------------------------------------------------------
run_model_suite() {
    local model_short="$1" card="$2" port="$3"
    local model extra_args
    model=$(resolve_model "${model_short}")
    extra_args=$(resolve_extra_args "${model_short}")

    echo ""
    echo "[${model_short}] ========================================"
    echo "[${model_short}]  Model: ${model}"
    echo "[${model_short}]  Card: ${card}  Port: ${port}"
    echo "[${model_short}] ========================================"

    for config in "${ALL_CONFIGS[@]}"; do
        echo ""
        echo "[${model_short}] --- Config: ${config} ---"

        if start_server "${model_short}" "${config}" "${model}" "${extra_args}" "${card}" "${port}"; then
            local kv_tokens
            kv_tokens=$(extract_kv_cache_tokens "${model_short}")
            echo "${kv_tokens}" > /tmp/bench_kv_${model_short}_${config}.txt
            echo "[${model_short}]     KV cache tokens: ${kv_tokens}"

            for scenario in "${SCENARIOS[@]}"; do
                run_scenario "${model_short}" "${config}" "${scenario}" "${model}" "${port}" || true
            done

            stop_server "${model_short}" "${port}"
        else
            echo "[${model_short}]     FAILED: ${config} server did not start"
            echo "FAIL" > /tmp/bench_kv_${model_short}_${config}.txt
            stop_server "${model_short}" "${port}"
        fi
    done
}

# ---------------------------------------------------------------------------
# print_model_summary — KV compression + per-scenario perf tables
# ---------------------------------------------------------------------------
print_model_summary() {
    local model_short="$1"
    local model
    model=$(resolve_model "${model_short}")

    echo ""
    echo "================================================================"
    echo "  SUMMARY: ${model} (${model_short})"
    echo "================================================================"

    # 1. KV Cache compression
    echo ""
    echo "--- KV Cache Compression ---"
    printf "  %-24s %14s %12s\n" "Config" "KV Cache Toks" "vs bf16"
    printf "  %-24s %14s %12s\n" "------------------------" "--------------" "------------"

    local bf16_tokens
    bf16_tokens=$(cat /tmp/bench_kv_${model_short}_bf16.txt 2>/dev/null || echo "N/A")

    for config in "${ALL_CONFIGS[@]}"; do
        local tokens
        tokens=$(cat /tmp/bench_kv_${model_short}_${config}.txt 2>/dev/null || echo "N/A")
        if [[ "${tokens}" == "FAIL" || "${tokens}" == "N/A" ]]; then
            printf "  %-24s %14s %12s\n" "${config}" "${tokens}" "-"
        elif [[ "${bf16_tokens}" != "N/A" && "${bf16_tokens}" != "0" ]]; then
            local ratio
            ratio=$(python3 -c "print(f'{${tokens}/${bf16_tokens}:.2f}x')")
            printf "  %-24s %14s %12s\n" "${config}" "${tokens}" "${ratio}"
        else
            printf "  %-24s %14s %12s\n" "${config}" "${tokens}" "-"
        fi
    done

    # 2. Per-scenario performance tables
    for scenario_spec in "${SCENARIOS[@]}"; do
        IFS=':' read -r name in_len out_len n_prompts conc <<< "${scenario_spec}"
        local total=$((in_len + out_len))
        if [[ ${total} -gt ${MAX_MODEL_LEN} ]]; then
            continue
        fi

        echo ""
        echo "--- ${name}  (ISL=${in_len}, OSL=${out_len}, N=${n_prompts}, C=${conc}) ---"
        printf "  %-22s %8s %10s %10s %10s %10s %10s %10s %10s\n" \
            "Config" "Req/s" "OutTok/s" "TTFT" "TPOT" "ITL" "p90 TTFT" "p90 TPOT" "p99 TTFT"
        printf "  %-22s %8s %10s %10s %10s %10s %10s %10s %10s\n" \
            "----------------------" "--------" "----------" "----------" "----------" "----------" "----------" "----------" "----------"

        for config in "${ALL_CONFIGS[@]}"; do
            local tag="${model_short}_${config}__${name}__${TIMESTAMP}"
            local json_file="${RESULT_DIR}/${tag}.json"

            local metrics
            metrics=$(docker exec ${CONTAINER} python3 -c "
import json, sys
try:
    with open('${json_file}') as f:
        d = json.load(f)
    req = d.get('request_throughput', 0)
    out = d.get('output_throughput', 0)
    ttft = d.get('mean_ttft_ms', 0)
    tpot = d.get('mean_tpot_ms', 0)
    itl = d.get('mean_itl_ms', 0)
    p90_ttft = d.get('p90_ttft_ms', 0)
    p90_tpot = d.get('p90_tpot_ms', 0)
    p99_ttft = d.get('p99_ttft_ms', 0)
    print(f'{req:.1f}|{out:.0f}|{ttft:.0f}|{tpot:.1f}|{itl:.1f}|{p90_ttft:.0f}|{p90_tpot:.1f}|{p99_ttft:.0f}')
except Exception:
    print('-|-|-|-|-|-|-|-')
" 2>/dev/null || echo "-|-|-|-|-|-|-|-")

            IFS='|' read -r req_s out_s ttft_s tpot_s itl_s p90ttft_s p90tpot_s p99ttft_s <<< "${metrics}"
            printf "  %-22s %8s %10s %10s %10s %10s %10s %10s %10s\n" \
                "${config}" "${req_s}" "${out_s}" "${ttft_s}" "${tpot_s}" "${itl_s}" "${p90ttft_s}" "${p90tpot_s}" "${p99ttft_s}"
        done
    done
}

# ===========================================================================
# Main
# ===========================================================================
echo "============================================"
echo "  TQ Performance Benchmark"
echo "  Models: ${MODEL_SHORTS[*]}"
echo "  Scenarios: ${#SCENARIOS[@]} (sliding_window suite)"
echo "  Configs: ${#ALL_CONFIGS[@]} per model"
echo "  max_model_len: ${MAX_MODEL_LEN}"
echo "  timestamp: ${TIMESTAMP}"
echo "============================================"

docker exec ${CONTAINER} mkdir -p ${RESULT_DIR}

# Launch each model in parallel, pinned to its own card + port
PIDS=()
for i in "${!MODEL_SHORTS[@]}"; do
    ms="${MODEL_SHORTS[$i]}"
    card="${i}"
    port=$((BASE_PORT + i))
    run_model_suite "${ms}" "${card}" "${port}" &
    PIDS+=($!)
done

# Wait for all parallel runs to finish
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || FAIL=1
done

# Print summaries (sequential, after all runs complete)
for ms in "${MODEL_SHORTS[@]}"; do
    print_model_summary "${ms}"
done

echo ""
echo "Theoretical KV compression: k8v4=2.6x, 4bit_nc=3.8x, k3v4_nc=~3.5x, 3bit_nc=4.9x"
echo "Logs: /tmp/bench_<model>_<config>__<scenario>__${TIMESTAMP}.log"
echo "Results: docker exec ${CONTAINER} ls ${RESULT_DIR}/"

exit ${FAIL}
