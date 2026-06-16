#!/bin/bash
# bench_tq_sla_concurrency.sh — SLA-bound max concurrency.
#
# Serving SLA: TTFT(p99) ≤ 5000 ms AND TPOT(p99) ≤ 200 ms.
# For each (model, config), find the highest concurrency that meets the SLA.
# Report concurrency multiplier vs BF16 per (model, preset).
#
# Per-model TP: TP=1 for 8B/E4B models, TP=2 for 14B.
# Scheduling: TP=1 models run in PARALLEL on separate GPUs (GPU 0, GPU 1),
# then TP=2 models run sequentially using both GPUs.
#
# Strategy: linear sweep with early-exit after 2 consecutive SLA failures.
#
# Output:
#   /workspace/bench_results/sla_sweep_<TS>.csv      — every measured point
#   /workspace/bench_results/sla_summary_<TS>.csv    — max conc passing + ratio
set -euo pipefail

CONTAINER="vllm-test"
BASE_PORT=8210
MAX_MODEL_LEN=4096
INPUT_LEN=1024
OUTPUT_LEN=512

SLA_TTFT_MS=5000
SLA_TPOT_MS=200

CONCURRENCIES=(1 2 4 8 16 32 64 128 256)
CONFIGS=("bf16" "turboquant_k8v4" "turboquant_4bit_nc" "turboquant_k3v4_nc" "turboquant_3bit_nc")

RESULT_DIR="/workspace/bench_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CSV_SWEEP="${RESULT_DIR}/sla_sweep_${TIMESTAMP}.csv"
CSV_SUMMARY="${RESULT_DIR}/sla_summary_${TIMESTAMP}.csv"

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
    local server_log="/tmp/vllm_sla_server_${model_short}.log"

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
    [[ ${n_prompts} -gt 512 ]] && n_prompts=512

    local tag="sla_${model_short}_${config}_c${conc}_${TIMESTAMP}"

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
        > /tmp/bench_${tag}.log 2>&1 || true

    docker exec -i ${CONTAINER} python3 - <<PY 2>/dev/null || echo "FAIL|FAIL|FAIL"
import json, os, sys
p = "${RESULT_DIR}/${tag}.json"
if not os.path.exists(p):
    print("FAIL|FAIL|FAIL"); sys.exit(0)
with open(p) as f:
    d = json.load(f)
tps = d.get('output_throughput', 0) or 0
if tps <= 0 or d.get('completed', 0) == 0:
    print("FAIL|FAIL|FAIL"); sys.exit(0)
print(f"{d.get('p99_ttft_ms',0):.2f}|{d.get('p99_tpot_ms',0):.2f}|{tps:.2f}")
PY
}

# ---------------------------------------------------------------------------
sweep_config() {
    local model_short="$1" config="$2" model="$3" tp_size="$4" gpu="$5" port="$6"

    if ! start_server "${model_short}" "${config}" "${model}" "${tp_size}" "${gpu}" "${port}"; then
        stop_server "${model_short}" "${port}"
        echo "0" > "/tmp/sla_max_${model_short}_${config}.txt"
        return
    fi

    local max_pass=0
    local consec_fail=0
    for conc in "${CONCURRENCIES[@]}"; do
        echo "[${model_short}]   ${config} conc=${conc}"
        local res
        res=$(run_point "${model_short}" "${config}" "${conc}" "${model}" "${port}")
        IFS='|' read -r p99_ttft p99_tpot out_tps <<< "${res}"

        local pass="0"
        if [[ "${p99_ttft}" != "FAIL" && -n "${p99_ttft}" ]]; then
            pass=$(python3 -c "
ttft=${p99_ttft}; tpot=${p99_tpot}; tps=${out_tps}
if tps <= 0 or (ttft == 0 and tpot == 0):
    print(0)
else:
    print(1 if (ttft <= ${SLA_TTFT_MS} and tpot <= ${SLA_TPOT_MS}) else 0)
")
        fi

        echo "${model_short},${config},${conc},${p99_ttft},${p99_tpot},${out_tps},${pass}" \
            >> "/tmp/sla_sweep_${TIMESTAMP}.csv"

        echo "[${model_short}]     p99_ttft=${p99_ttft}ms p99_tpot=${p99_tpot}ms out_tps=${out_tps} pass=${pass}"

        if [[ "${pass}" == "1" ]]; then
            max_pass="${conc}"
            consec_fail=0
        else
            consec_fail=$((consec_fail + 1))
            if [[ ${consec_fail} -ge 2 ]]; then
                echo "[${model_short}]   2 consecutive fails — stopping sweep"
                break
            fi
        fi
    done

    echo "${max_pass}" > "/tmp/sla_max_${model_short}_${config}.txt"
    stop_server "${model_short}" "${port}"
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
        sweep_config "${model_short}" "${config}" "${model}" "${tp_size}" "${gpu}" "${port}"
    done
}

# ---------------------------------------------------------------------------
emit_summary() {
    docker exec ${CONTAINER} mkdir -p "${RESULT_DIR}"

    docker cp "/tmp/sla_sweep_${TIMESTAMP}.csv" "${CONTAINER}:${CSV_SWEEP}" 2>/dev/null || true
    docker exec ${CONTAINER} bash -c \
        "(echo 'model,config,concurrency,p99_ttft_ms,p99_tpot_ms,output_tps,sla_pass'; cat ${CSV_SWEEP} 2>/dev/null) > ${CSV_SWEEP}.tmp && mv ${CSV_SWEEP}.tmp ${CSV_SWEEP}"

    docker exec ${CONTAINER} bash -c "echo 'model,config,max_conc_passing_sla,ratio_vs_bf16' > ${CSV_SUMMARY}"
    for ms in "${MODEL_SHORTS[@]}"; do
        local bf16_max
        bf16_max=$(cat "/tmp/sla_max_${ms}_bf16.txt" 2>/dev/null || echo "0")
        for config in "${CONFIGS[@]}"; do
            local m
            m=$(cat "/tmp/sla_max_${ms}_${config}.txt" 2>/dev/null || echo "0")
            local ratio="-"
            if [[ "${bf16_max}" != "0" ]]; then
                ratio=$(python3 -c "print(f'{${m}/${bf16_max}:.2f}x')")
            fi
            docker exec ${CONTAINER} bash -c \
                "echo '${ms},${config},${m},${ratio}' >> ${CSV_SUMMARY}"
        done
    done

    echo ""
    echo "================================================================"
    echo "  SLA: TTFT(p99) ≤ ${SLA_TTFT_MS}ms  AND  TPOT(p99) ≤ ${SLA_TPOT_MS}ms"
    echo "================================================================"
    echo ""
    echo "Per-point sweep: ${CSV_SWEEP}"
    docker exec ${CONTAINER} cat "${CSV_SWEEP}"
    echo ""
    echo "Summary: ${CSV_SUMMARY}"
    docker exec ${CONTAINER} cat "${CSV_SUMMARY}"
}

# ===========================================================================
echo "============================================"
echo "  SLA-bound Max Concurrency"
echo "  Models:   ${MODEL_SHORTS[*]}"
echo "  Configs:  ${CONFIGS[*]}"
echo "  SLA:      TTFT(p99)≤${SLA_TTFT_MS}ms, TPOT(p99)≤${SLA_TPOT_MS}ms"
echo "  Conc:     ${CONCURRENCIES[*]}"
echo "  ISL/OSL:  ${INPUT_LEN}/${OUTPUT_LEN}  max_model_len=${MAX_MODEL_LEN}"
echo "  Timestamp: ${TIMESTAMP}"
echo "  TP=1 models run in parallel on separate GPUs"
echo "============================================"

docker exec ${CONTAINER} mkdir -p ${RESULT_DIR}
: > "/tmp/sla_sweep_${TIMESTAMP}.csv"

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

emit_summary
exit ${FAIL}
