#!/usr/bin/env python3
"""Aggregate all 5 TQ benchmark categories into REPORT_TQ_FULL.md.

Reads the per-category result dirs under bench-results/:
  cat2_throughput/  (+ the earlier tq_perf/ for short/decode/mixed/high_load@4096)
  cat3_sla/sla_summary.csv, sla_sweep.csv
  cat4_longctx/longctx.csv
  cat5_ruler/ruler_summary.csv
Compares to the v0.22.1rc1 FP8 paper's published capacity. TQ-only measured.
"""
import csv
import glob
import json
import os

ROOT = "/home/intel/models/bench-results"
SCEN = ["short_decode", "decode_heavy", "mixed", "high_load",
        "long_prefill", "very_long_prefill"]

PAPER = {
    ("llama31", 1): {"name": "Llama-3.1-8B", "bf16": 141824, "fp8": 283584},
    ("deepseekr1", 1): {"name": "DeepSeek-R1-7B", "bf16": 328576, "fp8": 656000},
    ("gemma3", 1): {"name": "Gemma-3-1B", "bf16": 906975, "fp8": 1813950},
    ("qwen3", 1): {"name": "Qwen3-8B", "bf16": 123584, "fp8": 247232},
    ("gemma4", 2): {"name": "Gemma-4-E4B", "bf16": 652211, "fp8": 1304423},
    ("qwen25", 2): {"name": "Qwen2.5-14B", "bf16": 198720, "fp8": 397504},
    ("mistral", 2): {"name": "Mistral-Small-24B", "bf16": 177216, "fp8": 354432},
    ("deepseekr1_70b", 4): {"name": "DeepSeek-R1-70B", "bf16": 112000, "fp8": 224000},
    ("llama33_70b", 4): {"name": "Llama-3.3-70B", "bf16": 110336, "fp8": 224000},
    ("qwen25_72b", 4): {"name": "Qwen2.5-72B", "bf16": 102016, "fp8": 204032},
}
ORDER = [("llama31", 1), ("deepseekr1", 1), ("gemma3", 1), ("qwen3", 1),
         ("gemma4", 2), ("qwen25", 2), ("mistral", 2),
         ("deepseekr1_70b", 4), ("llama33_70b", 4), ("qwen25_72b", 4)]
NAME = {k: v["name"] for k, v in PAPER.items()}


def jget(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def latest(dirs, short, tp, scen):
    """Latest output_throughput for (short,tp,scen) across given result dirs."""
    best = None
    for d in dirs:
        for f in sorted(glob.glob(
                f"{ROOT}/{d}/{short}_turboquant_4bit_nc_tp{tp}__{scen}__*.json")):
            dd = jget(f)
            if dd:
                best = dd.get("output_throughput")
    return best


def kv_tokens(short, tp):
    for d in ("cat2_throughput", "tq_perf"):
        p = f"{ROOT}/{d}/logs/kv_{short}_turboquant_4bit_nc_tp{tp}.txt"
        if os.path.exists(p):
            v = open(p).read().strip()
            if v.isdigit():
                return int(v)
    return None


def main():
    out = ["# TurboQuant 4-bit — full 5-category benchmark vs FP8 paper (Intel Arc Pro B70)\n",
           "TQ-only measured (`turboquant_4bit_nc`, fp8 weights; Gemma-4 bf16 weights). "
           "BF16/FP8 columns are the v0.22.1rc1 paper's published values.\n"]

    # ---- Cat 1: capacity ----
    out.append("\n## Category 1 — KV-cache capacity (max_model_len=4096)\n")
    out.append("| Model | TP | BF16 (pub) | FP8 (pub) | TQ-4bit (meas) | TQ/BF16 | TQ/FP8 |")
    out.append("|---|--:|--:|--:|--:|--:|--:|")
    for k in ORDER:
        p = PAPER[k]; s, tp = k; tq = kv_tokens(s, tp)
        tqs = f"{tq:,}" if tq else "—"
        vb = f"{tq/p['bf16']:.2f}×" if tq else "—"
        vf = f"{tq/p['fp8']:.2f}×" if tq else "—"
        out.append(f"| {p['name']} | {tp} | {p['bf16']:,} | {p['fp8']:,} | {tqs} | {vb} | {vf} |")

    # ---- Cat 2: throughput (merge 4096 + 8192 runs) ----
    out.append("\n## Category 2 — Output throughput (tok/s), 6 scenarios\n")
    out.append("| Model | TP | " + " | ".join(SCEN) + " |")
    out.append("|---|--:|" + "|".join(["--:"]*len(SCEN)) + "|")
    dirs = ["cat2_throughput", "tq_perf"]
    for k in ORDER:
        s, tp = k
        cells = []
        for sc in SCEN:
            v = latest(dirs, s, tp, sc)
            cells.append(f"{v:.0f}" if v else "—")
        out.append(f"| {NAME[k]} | {tp} | " + " | ".join(cells) + " |")

    # ---- Cat 3: SLA max concurrency ----
    out.append("\n## Category 3 — SLA-bounded max concurrency (p99 TTFT≤5s, TPOT≤200ms)\n")
    sla = {}
    csvp = f"{ROOT}/cat3_sla/sla_summary.csv"
    if os.path.exists(csvp):
        for r in csv.DictReader(open(csvp)):
            sla[r["model"]] = (r.get("max_conc_pass", "—"), r.get("weights", ""))
    out.append("| Model | TP | max concurrency (TQ) | weights |")
    out.append("|---|--:|--:|:--|")
    for k in ORDER:
        p = PAPER[k]
        mc, w = sla.get(resolve_full(k), ("—", ""))
        out.append(f"| {p['name']} | {k[1]} | {mc} | {w} |")

    # ---- Cat 4: long-context peak throughput ----
    out.append("\n## Category 4 — Long-context peak output throughput (tok/s)\n")
    lc = {}  # (model, ctx) -> max out_tps
    csvp = f"{ROOT}/cat4_longctx/longctx.csv"
    if os.path.exists(csvp):
        for r in csv.DictReader(open(csvp)):
            try:
                v = float(r["out_tps"])
            except (ValueError, KeyError):
                continue
            key = (r["model"], r["ctx"])
            lc[key] = max(lc.get(key, 0), v)
    out.append("| Model | TP | peak@16K | peak@32K |")
    out.append("|---|--:|--:|--:|")
    for k in ORDER:
        p = PAPER[k]; full = resolve_full(k)
        v16 = lc.get((full, "16384")); v32 = lc.get((full, "32768"))
        s16 = f"{v16:.0f}" if v16 else "—"
        s32 = f"{v32:.0f}" if v32 else "—"
        out.append(f"| {p['name']} | {k[1]} | {s16} | {s32} |")

    # ---- Cat 5: RULER composite ----
    # CSV rows are: model,tp,config,weights,ctx,task,metric,score  (8 fields;
    # the cat5 writer emits an extra metric column, e.g. "...,niah_single_1:4096,none,0.52").
    # Composite = unweighted mean of all per-subtask scores at that ctx.
    out.append("\n## Category 5 — RULER accuracy (composite, TQ-4bit)\n")
    ruler = {}  # (model, ctx) -> [subtask scores]
    csvp = f"{ROOT}/cat5_ruler/ruler_summary.csv"
    if os.path.exists(csvp):
        for parts in csv.reader(open(csvp)):
            if len(parts) < 7 or parts[0] == "model":
                continue
            model, _tp, _cfg, _w, ctx, task = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            score = parts[-1]  # score is always the last field (7- or 8-col rows)
            if task in ("ALL",) or ":" not in task:
                continue  # skip EVAL_FAIL/OOM marker rows
            # task is "<subtask>:<seqlen>". lm_eval emits a stale ':4096' key (= -1
            # sentinel) alongside the real ':<ctx>' key when ctx!=4096. Keep only the
            # metric whose seqlen matches this run's ctx, and drop the -1 sentinels.
            try:
                seqlen = task.rsplit(":", 1)[1]
                val = float(score)
            except (ValueError, IndexError):
                continue
            if seqlen == ctx and val >= 0:
                ruler.setdefault((model, ctx), []).append(val)
    out.append("| Model | TP | 4K | 16K | 32K |")
    out.append("|---|--:|--:|--:|--:|")
    for k in ORDER:
        p = PAPER[k]; full = resolve_full(k)
        def comp(ctx):
            vals = ruler.get((full, ctx))
            return f"{sum(vals)/len(vals):.3f}" if vals else "—"
        out.append(f"| {p['name']} | {k[1]} | {comp('4096')} | {comp('16384')} | {comp('32768')} |")

    out.append("\n_'—' = skipped (ISL+OSL>maxlen), OOM, or not yet run. "
               "Cat-5 composite = unweighted mean of ruler sub-task scores in the CSV._\n")

    rep = "\n".join(out) + "\n"
    open(f"{ROOT}/REPORT_TQ_FULL.md", "w").write(rep)
    print(rep)
    print(f"\nWritten to {ROOT}/REPORT_TQ_FULL.md")


FULL_NAMES = {
    "llama31": "meta-llama/Llama-3.1-8B-Instruct",
    "deepseekr1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "gemma3": "google/gemma-3-1b-it",
    "qwen3": "Qwen/Qwen3-8B",
    "gemma4": "google/gemma-4-E4B-it",
    "qwen25": "Qwen/Qwen2.5-14B-Instruct",
    "mistral": "mistralai/Mistral-Small-24B-Instruct-2501",
    "llama33_70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen25_72b": "Qwen/Qwen2.5-72B-Instruct",
    "deepseekr1_70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}


def resolve_full(key):
    return FULL_NAMES[key[0]]


if __name__ == "__main__":
    main()
