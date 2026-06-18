#!/usr/bin/env bash
# Category 3 — SLA-bounded max concurrency, TQ-only.
# Concurrency ladder {1,2,4,8,16,32,64,128,256} at ISL=1024 OSL=512,
# max_model_len=4096. Gate: p99 TTFT <= 5000 ms AND p99 TPOT <= 200 ms.
# Early-exit after 2 consecutive SLA failures. Runs INSIDE the container.
#
# Usage:  bash cat3_sla.sh <model_short> [CARD]
set -uo pipefail
HERE="$(dirname "$0")"; source "$HERE/tq_common.sh"

MS="$1"; CARD="${2:-0}"
PORT=$((8210 + CARD))
MAXLEN=4096; ISL=1024; OSL=512
SLA_TTFT=5000; SLA_TPOT=200
LADDER=(1 2 4 8 16 32 64 128 256)
RES="$RESULT_ROOT/cat3_sla"; LOGD="$RES/logs"; DRV="$RES/driver_logs"
mkdir -p "$RES" "$LOGD" "$DRV"
TS="$(date +%Y%m%d_%H%M%S)"
exec > >(tee -a "$DRV/${MS}.log") 2>&1

MODEL="$(resolve_model "$MS")"; TP="$(resolve_tp "$MS")"
SERVER_LOG="$LOGD/server_${MS}_tp${TP}.log"
SUMMARY="$RES/sla_summary.csv"
[[ -f "$SUMMARY" ]] || echo "model,tp,config,weights,max_conc_pass" > "$SUMMARY"
SWEEP="$RES/sla_sweep.csv"
[[ -f "$SWEEP" ]] || echo "model,tp,config,conc,p99_ttft_ms,p99_tpot_ms,out_tps,pass" > "$SWEEP"
trap 'tq_cleanup "$PORT"' EXIT

echo "[cat3 ${MS} tp${TP}] start card=$CARD port=$PORT $(date)"
if ! tq_start_server "$MS" "$PORT" "$MAXLEN" 0.92 8192 "" "$SERVER_LOG"; then
  echo "[cat3 ${MS}] SERVER FAILED"; tail -25 "$SERVER_LOG"
  echo "${MODEL},${TP},${CONFIG},FAIL,0" >> "$SUMMARY"; exit 1
fi
echo "[cat3 ${MS}] healthy weights=$WEIGHTS_USED"
tq_warmup "$MODEL" "$PORT"

max_pass=0; consec_fail=0
for conc in "${LADDER[@]}"; do
  # Prompt count: full n=clamp(conc*4,100,512) for TP<=2. For TP=4 (70B) the
  # low-concurrency points are serial-bound (100 prompts @conc=1 ≈ 74min on a
  # 70B), so use a lighter n=clamp(conc*4,24,256): still enough requests for a
  # stable p99 SLA pass/fail decision, ~4x faster at low conc. Full ladder kept
  # so the SLA boundary (70B max ≈ conc 8) is still located exactly.
  if (( TP >= 4 )); then
    n=$((conc*4)); (( n<24 )) && n=24; (( n>256 )) && n=256
  else
    n=$((conc*4)); (( n<100 )) && n=100; (( n>512 )) && n=512
  fi
  tag="sla_${MS}_${CONFIG}_c${conc}_${TS}"
  tq_bench "$MODEL" "$PORT" "$ISL" "$OSL" "$n" "$conc" "$RES" "$tag" \
    concurrency="$conc" tp="$TP" > "$LOGD/${tag}.log" 2>&1 || true
  read p99t p99p tps <<< "$(python3 - "$RES/$tag.json" <<'PY'
import json,sys,os
p=sys.argv[1]
try:
    d=json.load(open(p))
    tps=d.get('output_throughput',0) or 0
    if tps<=0 or d.get('completed',0)==0: print("FAIL FAIL FAIL")
    else: print(f"{d.get('p99_ttft_ms',0):.2f} {d.get('p99_tpot_ms',0):.2f} {tps:.2f}")
except Exception: print("FAIL FAIL FAIL")
PY
)"
  pass=0
  if [[ "$p99t" != "FAIL" ]]; then
    pass=$(python3 -c "print(1 if (${p99t}<=${SLA_TTFT} and ${p99p}<=${SLA_TPOT}) else 0)")
  fi
  echo "${MODEL},${TP},${CONFIG},${conc},${p99t},${p99p},${tps},${pass}" >> "$SWEEP"
  echo "  conc=$conc p99_ttft=${p99t} p99_tpot=${p99p} tps=${tps} pass=${pass}"
  if [[ "$pass" == "1" ]]; then max_pass=$conc; consec_fail=0
  else consec_fail=$((consec_fail+1)); (( consec_fail>=2 )) && { echo "  2 consecutive fails — stop"; break; }; fi
done
echo "${MODEL},${TP},${CONFIG},${WEIGHTS_USED},${max_pass}" >> "$SUMMARY"
echo "[cat3 ${MS}] max_conc_pass=${max_pass} DONE $(date)"
