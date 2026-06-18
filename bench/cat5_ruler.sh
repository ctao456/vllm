#!/usr/bin/env bash
# Category 5 — RULER long-context accuracy, TQ-only.
# ctx {4096,16384,32768}; lm_eval 'ruler' task against the running vLLM server.
# Server: --max-num-seqs 2, max-num-batched-tokens=max(ctx,8192). num_concurrent=1.
# limit=25 for slow big models (TP>=2). Runs INSIDE container, one model at a time.
# Usage:  bash cat5_ruler.sh <model_short> [CARD]
set -uo pipefail
HERE="$(dirname "$0")"; source "$HERE/tq_common.sh"

MS="$1"; CARD="${2:-0}"
PORT=$((8230 + CARD))
CTXS=(4096 16384 32768)
RES="$RESULT_ROOT/cat5_ruler"; LOGD="$RES/logs"; DRV="$RES/driver_logs"
mkdir -p "$RES" "$LOGD" "$DRV"
TS="$(date +%Y%m%d_%H%M%S)"
exec > >(tee -a "$DRV/${MS}.log") 2>&1

MODEL="$(resolve_model "$MS")"; TP="$(resolve_tp "$MS")"
CSV="$RES/ruler_summary.csv"
[[ -f "$CSV" ]] || echo "model,tp,config,weights,ctx,task,score" > "$CSV"
# RULER sample limit: 50 for TP=1, 25 for the slower TP>=2 models (per user spec).
# Composite scores are robust to this; speeds up the TP=1 full-suite runs.
LIMIT="50"; [[ "$TP" -ge 2 ]] && LIMIT="25"

run_ruler() {  # ctx
  local ctx="$1"
  local outdir="$RES/ruler_${MS}_${CONFIG}_ctx${ctx}_${TS}"
  local elog="$LOGD/ruler_${MS}_ctx${ctx}.log"
  echo "    RULER ctx=$ctx limit=$LIMIT $(date +%H:%M:%S)"
  docker_python_ruler "$ctx" "$outdir" > "$elog" 2>&1
  return $?
}

docker_python_ruler() {  # ctx outdir
  local ctx="$1" outdir="$2"
  python3 - "$ctx" "$outdir" "$PORT" "$MODEL" "$MS" "$LIMIT" <<'PY'
import sys, json, os
ctx=int(sys.argv[1]); outdir=sys.argv[2]; port=sys.argv[3]; model=sys.argv[4]
ms=sys.argv[5]; limit=None if sys.argv[6]=="None" else int(sys.argv[6])
import lm_eval.tasks.ruler.common_utils as cu
cu.DEFAULT_SEQ_LENGTHS=[ctx]
from lm_eval import evaluator
# Explicit 12-task list = full RULER minus ruler_qa_hotpot, whose dataset host
# (curtis.ml.cmu.edu) returns 504 on this network. SQuAD QA is kept (reachable).
RULER_TASKS=['niah_single_1','niah_single_2','niah_single_3',
             'niah_multikey_1','niah_multikey_2','niah_multikey_3',
             'niah_multiquery','niah_multivalue','ruler_vt','ruler_cwe',
             'ruler_fwe','ruler_qa_squad']
model_args={'base_url':f'http://localhost:{port}/v1/completions','model':model,
            'tokenizer_backend':'huggingface','num_concurrent':1,'max_retries':5,'timeout':1200}
results=evaluator.simple_evaluate(model='local-completions', model_args=model_args,
    tasks=RULER_TASKS, metadata={'max_seq_lengths':[ctx],'tokenizer':model},
    batch_size='auto', log_samples=False, limit=limit)
os.makedirs(outdir, exist_ok=True)
json.dump(results.get('results',{}), open(os.path.join(outdir,'results.json'),'w'), indent=2, default=str)
print(json.dumps(results.get('results',{}), indent=2, default=str))
PY
}

echo "[cat5 ${MS} tp${TP}] start card=$CARD port=$PORT limit=$LIMIT $(date)"
for ctx in "${CTXS[@]}"; do
  maxb=$ctx; (( maxb<8192 )) && maxb=8192
  SERVER_LOG="$LOGD/server_${MS}_tp${TP}_ctx${ctx}.log"
  trap 'tq_cleanup "$PORT"' EXIT
  echo "  ctx=$ctx — starting server $(date +%H:%M:%S)"
  if ! tq_start_server "$MS" "$PORT" "$ctx" 0.92 "$maxb" 2 "$SERVER_LOG"; then
    echo "  ctx=$ctx SERVER FAILED/OOM"; tail -15 "$SERVER_LOG"
    echo "${MODEL},${TP},${CONFIG},${WEIGHTS_USED:-NA},${ctx},ALL,OOM_OR_FAIL" >> "$CSV"
    tq_cleanup "$PORT"; continue
  fi
  echo "  ctx=$ctx healthy weights=$WEIGHTS_USED"
  outdir="$RES/ruler_${MS}_${CONFIG}_ctx${ctx}_${TS}"
  if run_ruler "$ctx"; then
    # extract per-task scores from results.json into the CSV
    python3 - "$outdir/results.json" "$MODEL" "$TP" "$CONFIG" "$WEIGHTS_USED" "$ctx" "$CSV" <<'PY'
import json,sys
rj,model,tp,cfg,w,ctx,csv=sys.argv[1:8]
try: res=json.load(open(rj))
except Exception: res={}
import re
with open(csv,'a') as f:
    for task,metrics in res.items():
        for k,v in metrics.items():
            if isinstance(v,(int,float)) and not k.endswith('_stderr'):
                f.write(f"{model},{tp},{cfg},{w},{ctx},{task}:{k},{v}\n")
PY
    echo "  ctx=$ctx RULER done"
  else
    echo "  ctx=$ctx RULER FAILED"; tail -15 "$LOGD/ruler_${MS}_ctx${ctx}.log"
    echo "${MODEL},${TP},${CONFIG},${WEIGHTS_USED},${ctx},ALL,EVAL_FAIL" >> "$CSV"
  fi
  tq_cleanup "$PORT"
done
echo "[cat5 ${MS}] DONE $(date)"
