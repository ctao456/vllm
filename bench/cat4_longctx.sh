#!/usr/bin/env bash
# Category 4 — long-context throughput, TQ-only.
# Context {16384,32768}; concurrency {1,2,4,8,16,32,64}; ISL=ctx/4, OSL=ctx/8;
# max-model-len=ctx, max-num-batched-tokens=max(ctx,8192). Runs INSIDE container.
# One server per (model, ctx). Usage:  bash cat4_longctx.sh <model_short> [CARD]
set -uo pipefail
HERE="$(dirname "$0")"; source "$HERE/tq_common.sh"

MS="$1"; CARD="${2:-0}"
PORT=$((8220 + CARD))
CTXS=(16384 32768)
# Concurrency ladder. Paper uses {1,2,4,8,16,32,64} and reports peak throughput
# + the conc=8 operating point. For TP=4 (70B) the 32/64 points at 16K/32K with
# 8192-token prefills cost hours each and 70B peak throughput is reached by
# conc<=16, so cap the 70B ladder at 16 (keeps the conc=8 point + the peak).
LADDER=(1 2 4 8 16 32 64)
RES="$RESULT_ROOT/cat4_longctx"; LOGD="$RES/logs"; DRV="$RES/driver_logs"
mkdir -p "$RES" "$LOGD" "$DRV"
TS="$(date +%Y%m%d_%H%M%S)"
exec > >(tee -a "$DRV/${MS}.log") 2>&1

MODEL="$(resolve_model "$MS")"; TP="$(resolve_tp "$MS")"
# Cap 70B (TP=4) ladder at 16 (see note above); keep full ladder for TP<=2.
if (( TP >= 4 )); then LADDER=(1 2 4 8 16); fi
CSV="$RES/longctx.csv"
[[ -f "$CSV" ]] || echo "model,tp,config,weights,ctx,conc,isl,osl,n,req_tps,out_tps,p99_ttft_ms,p99_tpot_ms,status" > "$CSV"

echo "[cat4 ${MS} tp${TP}] start card=$CARD port=$PORT $(date)"
for ctx in "${CTXS[@]}"; do
  isl=$((ctx/4)); osl=$((ctx/8)); maxb=$ctx; (( maxb<8192 )) && maxb=8192
  SERVER_LOG="$LOGD/server_${MS}_tp${TP}_ctx${ctx}.log"
  trap 'tq_cleanup "$PORT"' EXIT
  echo "  ctx=$ctx isl=$isl osl=$osl — starting server $(date +%H:%M:%S)"
  if ! tq_start_server "$MS" "$PORT" "$ctx" 0.92 "$maxb" "" "$SERVER_LOG"; then
    echo "  ctx=$ctx SERVER FAILED/OOM"; tail -15 "$SERVER_LOG"
    echo "${MODEL},${TP},${CONFIG},${WEIGHTS_USED:-NA},${ctx},NA,${isl},${osl},NA,NA,NA,NA,NA,OOM_OR_FAIL" >> "$CSV"
    tq_cleanup "$PORT"; continue
  fi
  echo "  ctx=$ctx healthy weights=$WEIGHTS_USED"
  tq_warmup "$MODEL" "$PORT"
  for conc in "${LADDER[@]}"; do
    n=$((conc*2)); (( n<16 )) && n=16; (( n>256 )) && n=256
    tag="longctx_${MS}_${CONFIG}_ctx${ctx}_c${conc}_${TS}"
    echo "    ctx=$ctx conc=$conc n=$n $(date +%H:%M:%S)"
    tq_bench "$MODEL" "$PORT" "$isl" "$osl" "$n" "$conc" "$RES" "$tag" \
      context_length="$ctx" concurrency="$conc" isl="$isl" osl="$osl" tp="$TP" \
      > "$LOGD/${tag}.log" 2>&1
    read rt ot pt pp st <<< "$(python3 - "$RES/$tag.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(f"{d.get('request_throughput',0):.4f} {d.get('output_throughput',0):.2f} {d.get('p99_ttft_ms',0):.1f} {d.get('p99_tpot_ms',0):.2f} ok")
except Exception: print("NA NA NA NA FAIL")
PY
)"
    echo "${MODEL},${TP},${CONFIG},${WEIGHTS_USED},${ctx},${conc},${isl},${osl},${n},${rt},${ot},${pt},${pp},${st}" >> "$CSV"
    [[ "$st" == "FAIL" ]] && { echo "    conc=$conc bench FAILED — stop ladder for ctx=$ctx"; break; }
  done
  tq_cleanup "$PORT"
done
echo "[cat4 ${MS}] DONE $(date)"
