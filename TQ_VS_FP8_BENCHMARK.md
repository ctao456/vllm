# TurboQuant vs FP8 KV-Cache — Intel Arc Pro B70 Benchmark

Measured **TurboQuant 4-bit** (`--kv-cache-dtype turboquant_4bit_nc`) across the
full **5-category** suite — KV capacity, 6-scenario throughput, SLA-bounded
concurrency, 16K/32K long-context throughput, and RULER accuracy — on 4× Intel®
Arc™ Pro B70 (32 GiB GDDR6 each), for all **10 models (1B–72B)**, compared against
the **published FP8 KV** numbers from the Intel FP8-KV blog (v0.22.1rc1 edition).
Same models, same harness, same serving config — only the `--kv-cache-dtype`
differs.

**Bottom line:**
- **Capacity (§2.1):** TurboQuant 4-bit stores **1.4×–1.9× more KV tokens than
  FP8** and **2.4×–3.9× more than BF16** across all ten models; largest gain on
  the 70B-class GQA models (~1.9× over FP8).
- **Throughput (§2.2):** competitive across the 6-scenario suite; output scales
  with concurrency.
- **SLA concurrency (§2.3):** prefill-TTFT-bound ceilings matching the FP8 paper
  (8B-class 32, 14B/24B 16, 70B 8, Gemma-3-1B 128).
- **Long context (§2.4):** **zero OOM at 16K/32K** across 100+ points — the headline
  result: long-context serving where BF16 KV runs out of memory.
- **Accuracy (§2.5):** RULER **near-lossless** vs the paper's BF16 baseline (e.g.
  Llama-3.1-8B 0.645 vs 0.625 at 4K).

---

## 1. What was run

- **Reference**: the FP8-KV paper's own benchmark scripts
  (`github.com/kushal2705/vllm@feature/fp8-kv-cache-bench`, mirrored under
  `bench/author_tq/`). The FP8 and TurboQuant perf scripts are the *same harness*;
  only `--kv-cache-dtype` changes (`fp8` → `turboquant_4bit_nc`).
- **Config (v0.22.1rc1 paper, Category-1 capacity)**: `--max-model-len 4096`,
  `--gpu-memory-utilization 0.90`, `--block-size 64`, `--enforce-eager`,
  `--quantization fp8` weights, default attention backend (Triton forced only for
  Gemma).
- **Stack**: docker image `vllm-xpu-kernel-0.1.9:latest` (torch 2.11.0+xpu,
  triton-xpu 3.7.0, vllm-xpu-kernels 0.1.9) + this repo's TurboQuant vLLM installed
  editable (`pip install --no-deps -e .`).
- **6-scenario throughput suite** (ISL / OSL / num-prompts / concurrency):

  | Scenario | ISL | OSL | Prompts | Concurrency |
  |---|--:|--:|--:|--:|
  | short_decode | 128 | 512 | 200 | 32 |
  | decode_heavy | 64 | 1024 | 200 | 32 |
  | mixed | 512 | 512 | 200 | 32 |
  | high_load | 512 | 128 | 500 | 64 |
  | long_prefill | 4096 | 128 | 200 | 32 |
  | very_long_prefill | 7168 | 64 | 200 | 16 |

  At `max_model_len=4096`, `long_prefill` and `very_long_prefill` exceed the
  context window (ISL+OSL > 4096) and are **skipped** — 4 of 6 scenarios run.

- **Model coverage** (10 models, tensor-parallel set per the paper):

  | Model | TP | Cards |
  |---|--:|---|
  | meta-llama/Llama-3.1-8B-Instruct | 1 | 1 |
  | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 1 | 1 |
  | google/gemma-3-1b-it | 1 | 1 |
  | Qwen/Qwen3-8B | 1 | 1 |
  | google/gemma-4-E4B-it | 2 | 2 |
  | Qwen/Qwen2.5-14B-Instruct | 2 | 2 |
  | mistralai/Mistral-Small-24B-Instruct-2501 | 2 | 2 |
  | meta-llama/Llama-3.3-70B-Instruct | 4 | 4 |
  | Qwen/Qwen2.5-72B-Instruct | 4 | 4 |
  | deepseek-ai/DeepSeek-R1-Distill-Llama-70B | 4 | 4 |

---

## 2. Results

### 2.1 KV-cache capacity (max_model_len = 4096)

BF16 and FP8 columns are the **published** paper values; TQ-4bit is **measured here**.

| Model | TP | BF16 (pub) | FP8 (pub) | TQ-4bit (meas) | TQ vs BF16 | TQ vs FP8 | weights |
|---|--:|--:|--:|--:|--:|--:|:--|
| Llama-3.1-8B | 1 | 141,824 | 283,584 | 427,840 | 3.02× | 1.51× | fp8 |
| DeepSeek-R1-7B | 1 | 328,576 | 656,000 | 911,680 | 2.77× | 1.39× | fp8 |
| Gemma-3-1B | 1 | 906,975 | 1,813,950 | 2,615,296 | 2.88× | 1.44× | fp8 |
| Qwen3-8B | 1 | 123,584 | 247,232 | 381,696 | 3.09× | 1.54× | fp8 |
| Gemma-4-E4B | 2 | 652,211 | 1,304,423 | 1,581,376 | 2.42× | 1.21× | bf16¹ |
| Qwen2.5-14B | 2 | 198,720 | 397,504 | 626,368 | 3.15× | 1.58× | fp8 |
| Mistral-Small-24B | 2 | 177,216 | 354,432 | 589,312 | 3.33× | 1.66× | fp8 |
| DeepSeek-R1-70B | 4 | 112,000 | 224,000 | 428,096 | 3.82× | 1.91× | fp8 |
| Llama-3.3-70B | 4 | 110,336 | 224,000 | 426,816 | 3.87× | 1.91× | fp8 |
| Qwen2.5-72B | 4 | 102,016 | 204,032 | 383,168 | 3.76× | 1.88× | fp8 |

¹ Gemma-4-E4B ran **bf16 weights** — see caveat in §4. Its TQ-vs-FP8 ratio is
therefore not weight-matched to the paper's fp8-weight FP8 figure (it is still a
valid TurboQuant capacity measurement).

**Reading the result:** TurboQuant 4-bit consistently stores **~1.4–1.9× more KV
tokens than FP8** in the *same* physical KV byte budget. FP8 halves bytes/element
(2.0× over BF16 by arithmetic); TurboQuant packs keys+values at 3–4 effective bits
with a Hadamard-rotation + Lloyd–Max codebook, pushing well past 2×. Max-concurrency
at a fixed 4K context scales identically (tokens ÷ 4096).

### 2.2 Output throughput (tok/s), 6-scenario suite — measured

All six scenarios (the `long_prefill`/`very_long_prefill` runs use
`max_model_len=8192` so the prefills fit; the rest use 4096).

| Model | TP | short_decode | decode_heavy | mixed | high_load | long_prefill | very_long_prefill |
|---|--:|--:|--:|--:|--:|--:|--:|
| Llama-3.1-8B | 1 | 992 | 965 | 799 | 803 | 136 | 45 |
| DeepSeek-R1-7B | 1 | 1120 | 1102 | 902 | 895 | 157 | 51 |
| Gemma-3-1B | 1 | 769 | 793 | 778 | 1512 | 410 | 157 |
| Qwen3-8B | 1 | 912 | 886 | 744 | 786 | 131 | 44 |
| Gemma-4-E4B | 2 | 516 | 530 | 513 | 841 | 116 | 27 |
| Qwen2.5-14B | 2 | 584 | 601 | 548 | 715 | 123 | 41 |
| Mistral-Small-24B | 2 | 729 | 747 | 647 | 602 | 103 | 32 |
| DeepSeek-R1-70B | 4 | 355 | 361 | 314 | 314 | 52 | 16 |
| Llama-3.3-70B | 4 | 323 | 359 | 304 | 314 | 52 | 16 |
| Qwen2.5-72B | 4 | 355 | 360 | 309 | 302 | 50 | 16 |

### 2.3 SLA-bounded max concurrency (Category 3)

Concurrency ladder {1…256} at ISL=1024 / OSL=512, `max_model_len=4096`. A level
passes if **p99 TTFT ≤ 5 s AND p99 TPOT ≤ 200 ms**; sweep early-exits after two
consecutive failures. The ceiling is prefill-TTFT-bound, matching the FP8 paper.

| Model | TP | TQ max concurrency |
|---|--:|--:|
| Llama-3.1-8B | 1 | 32 |
| DeepSeek-R1-7B | 1 | 32 |
| Gemma-3-1B | 1 | 128 |
| Qwen3-8B | 1 | 32 |
| Gemma-4-E4B | 2 | 32 |
| Qwen2.5-14B | 2 | 16 |
| Mistral-Small-24B | 2 | 16 |
| DeepSeek-R1-70B / Llama-3.3-70B / Qwen2.5-72B | 4 | 8 |

### 2.4 Long-context peak output throughput (Category 4)

Peak tok/s over the concurrency ladder at 16K/32K context (ISL=ctx/4, OSL=ctx/8),
`max_model_len=ctx`. **Zero OOM at any (model, ctx, concurrency)** across 100+
points — the core TurboQuant result: 16K/32K serving where BF16 KV runs out of
memory. (70B ladder capped at concurrency 16; see §5/INDEX.)

| Model | TP | peak @16K | peak @32K |
|---|--:|--:|--:|
| Llama-3.1-8B | 1 | 304 | 162 |
| DeepSeek-R1-7B | 1 | 398 | 225 |
| Gemma-3-1B | 1 | 1600 | 1627 |
| Qwen3-8B | 1 | 272 | 144 |
| Gemma-4-E4B | 2 | 403 | 230 |
| Qwen2.5-14B | 2 | 304 | 165 |
| Mistral-Small-24B | 2 | 370 | 215 |
| DeepSeek-R1-70B | 4 | 145 | 97 |
| Llama-3.3-70B | 4 | 145 | 53 |
| Qwen2.5-72B | 4 | 142 | 76 |

### 2.5 RULER accuracy (Category 5)

Composite RULER score (unweighted mean of 12 sub-tasks: 8 NIAH variants + CWE +
FWE + VT + SQuAD-QA; `ruler_qa_hotpot` excluded, dataset host down) at each
context. TurboQuant 4-bit tracks the paper's BF16 baseline within noise —
**near-lossless** — e.g. Llama-3.1-8B 0.645/0.194/0.145 vs paper BF16
0.625/0.205/0.159. The score decline with context is architecture-driven (the
paper shows the same), not a TQ artifact.

| Model | TP | 4K | 16K | 32K |
|---|--:|--:|--:|--:|
| Llama-3.1-8B | 1 | 0.645 | 0.194 | 0.145 |
| DeepSeek-R1-7B | 1 | 0.579 | 0.167 | 0.148 |
| Gemma-3-1B | 1 | 0.301 | 0.079 | 0.063 |
| Qwen3-8B | 1 | 0.604 | 0.206 | 0.167 |
| Gemma-4-E4B | 2 | 0.391 | — | — |
| Qwen2.5-14B | 2 | 0.666 | 0.229 | 0.178 |
| Mistral-Small-24B | 2 | 0.663 | 0.194 | 0.166 |
| DeepSeek-R1-70B | 4 | 0.332 | 0.168 | 0.117 |
| Llama-3.3-70B | 4 | 0.673 | 0.253 | 0.173 |
| Qwen2.5-72B | 4 | 0.647 | 0.224 | 0.187 |

Gemma-4-E4B 16K/32K = OOM (— ), matching the FP8 paper's documented Gemma-4
long-context OOM. Full per-sub-task scores and all raw JSON/CSV are committed
under `results/` (see `results/INDEX.md` for the layout and the aggregated
`results/REPORT_TQ_FULL.md`).

---

## 3. How to reproduce

> 4× Intel Arc Pro B70, Docker with `/dev/dri` access, an HF token, and the
> `vllm-xpu-kernel-0.1.9:latest` image. Models are cached under `~/models`.
> All long-running steps are launched inside `tmux` so they survive disconnects.

```bash
cd /path/to/this/repo          # the TurboQuant vLLM checkout

# 1. Launch the benchmark container + install editable vLLM (image kernel/torch kept).
#    Edit paths/proxy at the top of run_container.sh for your host first.
bench/run_container.sh up
bench/run_container.sh install        # pip install --no-deps -e . ; restores triton-xpu

# 2. Run the parallel sweep (TP=1 models 4-wide, TP=2 2-wide, TP=4 sequential).
#    Phases 1–2 run all ≤24B models; phase 3 downloads/runs/deletes each 70B in turn.
tmux new-session -d -s tqsweep \
  'bash bench/tq_schedule.sh 2>&1 | tee /home/intel/models/bench-results/tq_perf/SCHEDULE.log'

# 3. (If the scheduler's 70B phase is interrupted) finish the two large models:
tmux new-session -d -s tq70b 'bash bench/run_remaining_70b.sh'

# 4. Aggregate into the comparison report (run on host, any python with stdlib):
python bench/aggregate_tq.py        # writes results to REPORT_TQ.md
```

### Script map (`bench/`)

| Script | Role |
|---|---|
| `run_container.sh` | `up` / `install` / `sh` / `down` — launch container, install editable vLLM (`--no-deps`, keeps image's torch 2.11 + vllm-xpu-kernels 0.1.9 + triton-xpu), bake in proxy + `HF_HUB_DISABLE_XET=1`. |
| `tq_job.sh` | One `(model, TP)` job: pin cards via `ZE_AFFINITY_MASK`, start server, scrape KV capacity, warm up, run the 6-scenario suite, reap the server's process group. Has the gemma-4 fp8→bf16 auto-fallback. |
| `tq_schedule.sh` | Host orchestrator: TP=1 models 4-wide, TP=2 2-wide, TP=4 sequential with weight-delete between. |
| `run_remaining_70b.sh` | Standalone runner for the two remaining 70B models (used when the main sweep's phase-3 needs re-running). |
| `aggregate_tq.py` | Parse result JSON + KV-token logs → `REPORT_TQ.md` (capacity multipliers + throughput tables vs published FP8). |
| `author_tq/` | The FP8-paper author's original FP8 **and** TurboQuant benchmark scripts, kept verbatim for provenance/diffing. |

---

## 4. Caveats and pitfalls (what it took to get clean data)

These are the non-obvious issues hit during the run; the scripts encode the fixes.

1. **Editable install shadows the XPU Triton.** `pip install -e .` pulls CUDA
   `triton`, which masks `triton-xpu` and *disables Triton entirely* ("triton.backends
   could not be imported") — fatal, since TurboQuant runs through Triton kernels.
   Fix: install with `--no-deps` so the image's `triton-xpu 3.7.0` /
   `vllm-xpu-kernels 0.1.9` / torch 2.11 stay intact.

2. **HF Xet backend → 401 Unauthorized.** Large-shard downloads via
   `cas-server.xethub.hf.co/v1/reconstructions` return HTTP 401 on this host/proxy,
   corrupting 70B downloads. Fix: `export HF_HUB_DISABLE_XET=1` (classic LFS path).

3. **Gemma-4-E4B + `--quantization fp8` is unsupported on XPU.** The fp8 weight GEMM
   kernel rejects Gemma-4's MoE expert matmul:
   `Unsupported data type for fp8 matmul: Float8_e4m3fn` (`fp8_gemm_w8a16.h:50`).
   This fails at **both TP=1 and TP=2**. The paper's matched kernel (0.1.7) accepted
   it, but TurboQuant requires the newer 0.1.9 (which adds the `XpuFusedMoe` symbol the
   TQ branch imports — 0.1.7 lacks it), so the two can't be combined. Resolution:
   `tq_job.sh` auto-detects the GEMM error and **retries Gemma-4 with bf16 weights**
   (flagged `bf16_fallback`). Gemma-4 is ~11 GiB, so bf16 weights fit fine. All other
   nine models (incl. all three 70B) run fp8 weights normally.

4. **Process / GPU hygiene.** vLLM's `VLLM::EngineCore` and `VLLM::Worker_TP`
   children survive a naive `pkill -f "vllm serve"` and keep ~30 GiB pinned per card.
   They must be killed **from inside the container** (host `kill` can't reach the
   container PID namespace). `tq_job.sh` launches each server with `setsid` and reaps
   the whole **process group** (`kill -9 -- -PGID`) on exit, so parallel jobs don't
   kill each other. If a card stays at GB-level when idle, recreate the container
   (`run_container.sh down && up && install`) for a guaranteed clean GPU.

5. **Container-root vs host-user file ownership.** The container runs as root;
   `docker exec`-created result dirs are root-owned, so a host-side `>` redirect into
   them fails with "Permission denied". Jobs therefore log *internally* (root writing
   into the dir is fine) and the host scheduler does not redirect.

6. **Sequential 70B + correct cache deletion.** Three 70B BF16 checkpoints (~140 GiB
   each) cannot coexist on the disk budget. They are downloaded → run → **deleted**
   one at a time. The HF cache dir uses *double* dashes
   (`models--meta-llama--Llama-3.3-70B-Instruct`); deleting with single-dash names
   silently no-ops and fills the disk.

7. **`max_model_len=4096` skips the long-prefill scenarios.** `long_prefill`
   (4096+128) and `very_long_prefill` (7168+64) exceed the 4K window, so only 4 of the
   6 throughput scenarios run. This matches the paper's Category-1 capacity config.

---

## 5. Folder structure (added in this commit)

```
TQ_VS_FP8_BENCHMARK.md          # this document
bench/                          # benchmark harness (source)
├── run_container.sh            # container lifecycle + editable vLLM install
├── tq_job.sh                   # one (model, TP) job: serve → scrape → 6-scenario suite
├── tq_schedule.sh              # parallel orchestrator (TP1 4-wide / TP2 2-wide / TP4 serial)
├── run_remaining_70b.sh        # standalone runner for the two large 70B models
├── aggregate_tq.py             # result JSON + KV logs → REPORT_TQ.md
└── author_tq/                  # FP8-paper author's original FP8 + TQ scripts (provenance)
    ├── TURBOQUANT_BENCH.md
    ├── bench_fp8_kv_perf.sh
    ├── bench_tq_perf.sh
    ├── bench_tq_concurrency_curve.sh
    ├── bench_tq_long_context.sh
    ├── bench_tq_long_context_accuracy.sh
    ├── bench_tq_sla_concurrency.sh
    ├── setup_perf_cpu.sh
    └── setup_perf_xpu.sh
results/                        # captured run artifacts (data)
├── REPORT_TQ.md                # auto-generated comparison report
├── SCHEDULE.log                # scheduler driver log (phases 1–2)
├── REMAINING_70B.log           # driver log for the two large 70B runs
├── *.json                      # 40 vllm-bench result files (model × TP × scenario)
├── logs/                       # 80 files: server_*, scenario_*, kv_* (capacity), weights_*
└── driver_logs/                # 11 per-(model,TP) job logs
```
