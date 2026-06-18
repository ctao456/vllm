#!/usr/bin/env bash
# Category 2 — 6-scenario throughput suite, TQ-only, UNBOUNDED context.
# Runs INSIDE the container; one model at a time (arg) on cards from $CARD.
#
# Uses max_model_len=8192 (FP8_KV_BENCH perf config) so ALL six scenarios run,
# including long_prefill (4096+128) and very_long_prefill (7168+64) which were
# skipped under the 4096 capacity config. Results -> bench-results/cat2_throughput.
#
# Usage (inside container):  bash cat2_throughput.sh <model_short> [CARD]
set -uo pipefail
HERE="$(dirname "$0")"; source "$HERE/tq_common.sh"

MS="$1"; CARD="${2:-0}"
PORT=$((8192 + CARD))
MAXLEN=8192
RES="$RESULT_ROOT/cat2_throughput"; LOGD="$RES/logs"; DRV="$RES/driver_logs"
mkdir -p "$RES" "$LOGD" "$DRV"
TS="$(date +%Y%m%d_%H%M%S)"
exec > >(tee -a "$DRV/${MS}.log") 2>&1

# Only the two scenarios missing from the 4096 run (long/very_long_prefill).
# To re-run the full suite, set FULL=1.
if [[ "${FULL:-0}" == "1" ]]; then
  SCEN=( "short_decode:128:512:200:32" "decode_heavy:64:1024:200:32" "mixed:512:512:200:32"
         "high_load:512:128:500:64" "long_prefill:4096:128:200:32" "very_long_prefill:7168:64:200:16" )
else
  SCEN=( "long_prefill:4096:128:200:32" "very_long_prefill:7168:64:200:16" )
fi

MODEL="$(resolve_model "$MS")"; TP="$(resolve_tp "$MS")"
SERVER_LOG="$LOGD/server_${MS}_tp${TP}.log"
trap 'tq_cleanup "$PORT"' EXIT

echo "[cat2 ${MS} tp${TP}] start card=$CARD port=$PORT maxlen=$MAXLEN $(date)"
if ! tq_start_server "$MS" "$PORT" "$MAXLEN" 0.92 8192 "" "$SERVER_LOG"; then
  echo "[cat2 ${MS}] SERVER FAILED"; tail -25 "$SERVER_LOG"; exit 1
fi
read kv mc < <(tq_capacity "$SERVER_LOG")
echo "[cat2 ${MS}] KV=$kv conc=${mc}x weights=$WEIGHTS_USED"
echo "$kv" > "$LOGD/kv_${MS}_tp${TP}.txt"
echo "$WEIGHTS_USED" > "$LOGD/weights_${MS}_tp${TP}.txt"

tq_warmup "$MODEL" "$PORT"
for spec in "${SCEN[@]}"; do
  IFS=':' read -r name isl osl n conc <<< "$spec"
  if (( isl + osl > MAXLEN )); then echo "  [$name] SKIP (>$MAXLEN)"; continue; fi
  tag="${MS}_${CONFIG}_tp${TP}__${name}__${TS}"
  echo "  [$name] isl=$isl osl=$osl n=$n conc=$conc $(date +%H:%M:%S)"
  tq_bench "$MODEL" "$PORT" "$isl" "$osl" "$n" "$conc" "$RES" "$tag" \
    scenario="$name" tp="$TP" > "$LOGD/scenario_${tag}.log" 2>&1 \
    && echo "  [$name] done" || echo "  [$name] FAILED"
done
echo "[cat2 ${MS}] DONE $(date)"
