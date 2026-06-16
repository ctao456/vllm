#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate the TQ-only sweep into a report vs the FP8 paper's published numbers.

Reads:
  tq_perf/<short>_turboquant_4bit_nc_tp<TP>__<scenario>__<ts>.json  (vllm bench)
  tq_perf/logs/kv_<short>_turboquant_4bit_nc_tp<TP>.txt             (KV tokens)
  tq_perf/logs/weights_<short>_..._tp<TP>.txt                        (bf16_fallback flag)

Writes tq_perf/REPORT_TQ.md. Picks the LATEST json per (short,tp,scenario).
Compares measured TQ-4bit capacity to the v0.22.1 FP8 paper's published
BF16 / FP8 KV token counts (FP8 = 2x BF16).
"""

import glob
import json
import os

RES = "/home/intel/models/bench-results/tq_perf"
LOGS = f"{RES}/logs"
SCEN_ORDER = [
    "short_decode",
    "decode_heavy",
    "mixed",
    "high_load",
    "long_prefill",
    "very_long_prefill",
]

# v0.22.1 paper Table 6 — published capacity (BF16 KV tokens, FP8 = 2x), per model+TP.
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
ORDER = [
    ("llama31", 1),
    ("deepseekr1", 1),
    ("gemma3", 1),
    ("qwen3", 1),
    ("gemma4", 2),
    ("qwen25", 2),
    ("mistral", 2),
    ("deepseekr1_70b", 4),
    ("llama33_70b", 4),
    ("qwen25_72b", 4),
]


def latest_json(short, tp, scen):
    pat = f"{RES}/{short}_turboquant_4bit_nc_tp{tp}__{scen}__*.json"
    files = sorted(glob.glob(pat))  # ts is sortable yyyymmdd_hhmmss
    if not files:
        return None
    try:
        return json.load(open(files[-1]))
    except Exception:
        return None


def kv_tokens(short, tp):
    try:
        v = open(f"{LOGS}/kv_{short}_turboquant_4bit_nc_tp{tp}.txt").read().strip()
        return int(v) if v.isdigit() else None
    except Exception:
        return None


def weights_note(short, tp):
    p = f"{LOGS}/weights_{short}_turboquant_4bit_nc_tp{tp}.txt"
    if os.path.exists(p):
        return open(p).read().strip()
    return "fp8"


def main():
    out = [
        "# TurboQuant (4bit_nc) on Intel Arc Pro B70 — measured vs FP8 paper\n",
        "Config: v0.22.1rc1 paper settings (max_model_len=4096, gpu-util=0.90, "
        "fp8 weights, block-size 64, enforce-eager). TQ KV cache = "
        "`turboquant_4bit_nc`. FP8/BF16 columns are the paper's **published** "
        "capacity; TQ tokens are **measured** here.\n",
    ]

    # ---- capacity ----
    out.append("\n## 1. KV-cache capacity (max_model_len=4096)\n")
    out.append(
        "| Model | TP | BF16 (pub) | FP8 (pub) | TQ-4bit (meas) | TQ vs BF16 | TQ vs FP8 | weights |"
    )
    out.append("|---|--:|--:|--:|--:|--:|--:|:--|")
    for key in ORDER:
        p = PAPER[key]
        short, tp = key
        tq = kv_tokens(short, tp)
        w = weights_note(short, tp)
        tq_s = f"{tq:,}" if tq else "—"
        vb = f"{tq / p['bf16']:.2f}×" if tq else "—"
        vf = f"{tq / p['fp8']:.2f}×" if tq else "—"
        out.append(
            f"| {p['name']} | {tp} | {p['bf16']:,} | {p['fp8']:,} | "
            f"{tq_s} | {vb} | {vf} | {w} |"
        )

    # ---- throughput ----
    out.append("\n## 2. Output throughput (tok/s), TQ-4bit — measured\n")
    out.append("| Model | TP | " + " | ".join(SCEN_ORDER) + " |")
    out.append("|---|--:|" + "|".join(["--:"] * len(SCEN_ORDER)) + "|")
    for key in ORDER:
        p = PAPER[key]
        short, tp = key
        cells = []
        for s in SCEN_ORDER:
            d = latest_json(short, tp, s)
            cells.append(f"{d['output_throughput']:.0f}" if d else "—")
        out.append(f"| {p['name']} | {tp} | " + " | ".join(cells) + " |")

    out.append(
        "\n_'—' in throughput = scenario skipped (ISL+OSL>4096) or not yet run._"
    )
    out.append(
        "_Gemma-4 ran bf16 weights (fp8 weight-quant unsupported by XPU "
        "fp8 GEMM kernel for its MoE architecture); all others fp8 weights._\n"
    )

    rep = "\n".join(out) + "\n"
    open(f"{RES}/REPORT_TQ.md", "w").write(rep)
    print(rep)
    print(f"\nWritten to {RES}/REPORT_TQ.md")


if __name__ == "__main__":
    main()
