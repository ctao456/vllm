# TurboQuant 5-Category Benchmark — Results Index

TQ-only (`--kv-cache-dtype turboquant_4bit_nc`) measurements on 4× Intel Arc Pro
B70, to compare against the published FP8-KV numbers in the v0.22.1rc1 paper
(`fp8_kv_b70_v0.22.1rc1_COMPLETE.pdf`). Same harness, same models, same serving
config; only the KV-cache dtype differs. See `../TQ_VS_FP8_BENCHMARK.md` for the
methodology, reproduction steps, and caveats.

## Layout

| Path | Category | Contents |
|---|---|---|
| `REPORT_TQ_FULL.md` | all | Aggregated tables (capacity, throughput, SLA, long-ctx, RULER) |
| `cat1_2_capacity_throughput/from_tq_perf/` | Cat 1 + 2 | KV capacity (max_model_len=4096) + short/decode/mixed/high_load throughput JSONs; `logs/kv_*.txt` hold the scraped KV-token counts |
| `cat1_2_capacity_throughput/from_cat2_8192/` | Cat 2 | long_prefill + very_long_prefill throughput (max_model_len=8192, unbounded) |
| `cat3_sla/` | Cat 3 | SLA concurrency sweep JSONs; `sla_summary.csv` (max conc passing p99 TTFT≤5s ∧ TPOT≤200ms), `sla_sweep.csv` (every point) |
| `cat4_longctx/` | Cat 4 | 16K/32K long-context throughput JSONs; `longctx.csv` (per ctx×conc: req/out tps, p99 ttft/tpot, status) |
| `cat5_ruler/` | Cat 5 | RULER accuracy per `ruler_<model>_<cfg>_ctx<N>/results.json`; `ruler_summary.csv` (per-task scores) |
| `orchestration_logs/` | — | Master scheduler logs (MASTER2_*, REMAINING_*) |

## Config (matches the v0.22.1rc1 paper)

- max_model_len: 4096 for Cat 1/3; 8192 for Cat 2 (so long/very-long prefills run); =ctx for Cat 4/5
- gpu-mem-util 0.92 (0.90 for Cat 1/2 capacity), block-size 64, enforce-eager
- `--quantization fp8` weights on all models **except Gemma-4** (XPU fp8-GEMM
  unsupported for its MoE → bf16 weights, flagged `bf16_fallback`)
- Cat 5 RULER: 12 of 13 sub-tasks (excludes `ruler_qa_hotpot` — its dataset host
  returned 504; SQuAD QA retained), `num_concurrent=1`, limit 50 (TP=1) / 25 (TP≥2)

## Trims applied to the 70B (TP=4) track (kept comparable to the paper)

- **Cat 3**: full concurrency ladder retained (the paper's 70B SLA ceiling is
  conc≈8, so all of 1→16 are still tested); only the per-point prompt count was
  reduced to `clamp(conc*4,24,256)` for stable-but-faster p99 at low concurrency.
- **Cat 4**: 70B ladder capped at concurrency 16 (paper reports the conc=8
  operating point + peak throughput, which 70B reach by ≤16); the slow 32/64
  points at 16K/32K with 8192-token prefills are skipped for 70B only.
- Small models (TP≤2) ran the full ladders unchanged.
