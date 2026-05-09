# TurboQuant Benchmark Reproduction Guide

Reproduces four benchmark suites that validate TurboQuant KV-cache compression
on Intel B-series GPUs (B70) under vLLM XPU.

| Script | What it measures |
|---|---|
| `bench_tq_perf.sh` | BF16 vs all TQ presets across 6 serving scenarios |
| `bench_tq_sla_concurrency.sh` | Max concurrency while meeting TTFT/TPOT SLA targets |
| `bench_tq_long_context.sh` | KV-starved concurrency at 16K / 32K context |
| `bench_tq_long_context_accuracy.sh` | RULER accuracy at 16K / 32K context (BF16 vs TQ) |

---

## Prerequisites

- Docker installed on the host
- Intel GPU with `/dev/dri` exposed to the container
- Internet access (or an HTTP proxy)

---

## Setup

### 1. Launch the Docker container

The benchmarks run from the **host** via `docker exec` into a named container.
The container must be called `vllm-test`.

```bash
docker run --rm -it --privileged --network=host --ipc=host \
  -e http_proxy=http://proxy-dmz.intel.com:912 \
  -e https_proxy=http://proxy-dmz.intel.com:912 \
  -e no_proxy=10.0.0.0/8,habana-labs.com,.habana-labs.com,intel.com,.intel.com,127.0.0.1,localhost \
  -e HF_HOME=/tmp \
  -e HF_TOKEN=<your_hf_token> \
  -v $(pwd):$(pwd) -w $(pwd) \
  -v ~/LLM:/tmp \
  -v /dev/dri/by-path:/dev/dri/by-path \
  --name=vllm-test \
  --device /dev/dri:/dev/dri \
  --entrypoint=/bin/bash \
  intel/vllm:0.17.0-xpu
```

### 2. Mark the repository as safe for Git

```bash
git config --global --add safe.directory .
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

Never mind the warning, stick to latest `triton-xpu` for best performance on XPU.

### 4. Build and install vLLM XPU backend

```bash
VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v
```

### 5. Install the latest vllm_xpu_kernels and Triton package for Intel XPU

The default triton package (for NVIDIA GPUs) may be installed as a transitive
dependency (e.g., via xgrammar). For Intel XPU, replace it with `triton-xpu`.

```bash
wget https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.6/vllm_xpu_kernels-0.1.6-cp38-abi3-manylinux_2_28_x86_64.whl
pip install vllm_xpu_kernels-0.1.6-cp38-abi3-manylinux_2_28_x86_64.whl

pip uninstall -y triton triton-xpu
pip install triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu
```

### 6. Upgrade transformers for Gemma 4 (inside container, once)

```bash
echo ">>> Upgrading transformers to HuggingFace HEAD ..."
uv pip uninstall transformers
uv pip install "git+https://github.com/huggingface/transformers"

echo ">>> Done. transformers version:"
python -c "import transformers; print(transformers.__version__)"
```

---

## Quality Test

Verify that TurboQuant KV cache compression works end-to-end:

> **Parameter notes vs. a standard CUDA run:**
>
> - `max_model_len=2048` — kept conservative to fit within XPU memory headroom during initial validation.
> - `gpu_memory_utilization=0.95` — XPU memory profiling is less conservative than CUDA; a higher fraction is needed to actually allocate a useful KV cache.
> - `enforce_eager=True` — disables `torch.compile` and XPU graph capture, which are not yet stable for TurboQuant on XPU. Without this flag the engine falls back anyway (see warnings in sample output below), but setting it explicitly silences spurious warnings.

```bash
cat > /tmp/test_tq.py << 'EOF'
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM("Qwen/Qwen2.5-7B-Instruct",
        kv_cache_dtype="turboquant_k3v4_nc",
        max_model_len=2048,
        gpu_memory_utilization=0.95,
        enforce_eager=True)
    for o in llm.generate(
        [
            "What is 2+2?",
            "Explain gravity in 3 sentences.",
            "Write a haiku about the moon.",
        ],
        SamplingParams(max_tokens=100),
    ):
        print(o.outputs[0].text[:200])
        print()
EOF
python /tmp/test_tq.py
```

---

## (Optional) Host-side Performance Tuning

For reproducible benchmark numbers, lock GPU frequency and set CPU governor to
`performance`. Run these **on the host** (not inside the container):

```bash
# Lock GPU frequency — pick a value appropriate for your B70 (e.g. 2200 MHz)
sudo bash setup_perf_xpu.sh 2200

# CPU only (called automatically by setup_perf_xpu.sh, but can be run standalone)
sudo bash setup_perf_cpu.sh
```

---

## (Optional) Offline Accuracy Sanity Checks

Quick offline smoke tests that compare BF16 vs TQ on 10 factual / math /
reasoning prompts. Run inside the container or via `docker exec`:

```bash
# Qwen3-8B — full attention, straightforward TQ candidate
python test_qwen38b_acc.py          # bf16 baseline
python test_qwen38b_acc.py --tq     # TurboQuant k3v4_nc

# Gemma 3-1B — sliding + global attention layers (text-only variant)
python test_gemma3_acc.py           # bf16 baseline
python test_gemma3_acc.py --tq      # TurboQuant k3v4_nc

# Gemma 4-E4B — hybrid attention, YOCO KV sharing, requires TRITON_ATTN backend
python test_gemma4e4b_acc.py        # TurboQuant k3v4_nc (default)
python test_gemma4e4b_acc.py --no-tq  # bf16 baseline
```

All three scripts exit `0` when all 10 checks pass and `1` otherwise.

---

## Benchmark Runs

All four scripts run from the **host**. They start/stop vLLM servers inside the
`vllm-test` container automatically. Results land in
`/workspace/bench_results/` inside the container.

### Test 1 — Performance: BF16 vs TQ presets across 6 scenarios

Compares `bf16` against `turboquant_k8v4`, `turboquant_4bit_nc`,
`turboquant_k3v4_nc`, and `turboquant_3bit_nc` across six fixed scenarios:

| Scenario | ISL | OSL | Prompts | Concurrency |
|---|---|---|---|---|
| short_decode | 128 | 512 | 200 | 32 |
| long_prefill | 4096 | 128 | 200 | 32 |
| mixed | 512 | 512 | 200 | 32 |
| high_load | 512 | 128 | 500 | 64 |
| very_long_prefill | 7168 | 64 | 200 | 16 |
| decode_heavy | 64 | 1024 | 200 | 32 |

`max_model_len=8192`. Models can be run in parallel, each pinned to a separate
GPU card.

```bash
# Single model on card 0
bash bench_tq_perf.sh qwen3

# Two models in parallel (qwen3 → card 0, gemma3 → card 1)
bash bench_tq_perf.sh qwen3 gemma3

# Two models in parallel (gemma3 → card 0, gemma4 → card 1)
bash bench_tq_perf.sh gemma3 gemma4
```

Available model aliases: `qwen3` (Qwen/Qwen3-8B), `gemma3` (google/gemma-3-1b-it),
`gemma4` (google/gemma-4-E4B-it).

Output: per-scenario tables with request/s, output tokens/s, TTFT, TPOT, ITL,
p90 and p99 percentiles, plus a KV cache compression ratio summary.
JSON result files: `docker exec vllm-test ls /workspace/bench_results/`.

---

### Test 2 — SLA-bound max concurrency

Finds the highest concurrency where **TTFT(p99) ≤ 5000 ms** and **TPOT(p99) ≤ 200 ms**
hold simultaneously. Sweeps `{1,2,4,8,16,32,64,128,256}` with early exit after
two consecutive failures. `max_model_len=4096`, ISL=1024, OSL=512.

TP=1 models (`gemma4_e4b`, `llama31_8b`) run in parallel on separate GPUs;
TP=2 models (`qwen25_14b`) run sequentially using both GPUs.

```bash
# All three models
bash bench_tq_sla_concurrency.sh

# Single model
bash bench_tq_sla_concurrency.sh llama31_8b

# Two specific models
bash bench_tq_sla_concurrency.sh gemma4_e4b qwen25_14b
```

Available model aliases: `gemma4_e4b` (google/gemma-4-E4B-it),
`llama31_8b` (meta-llama/Llama-3.1-8B-Instruct),
`qwen25_14b` (Qwen/Qwen2.5-14B-Instruct).

Output:

- `/workspace/bench_results/sla_sweep_<TS>.csv` — every measured concurrency point
- `/workspace/bench_results/sla_summary_<TS>.csv` — max passing concurrency + ratio vs BF16

---

### Test 3 — Long-context capacity (16K / 32K)

Demonstrates that TurboQuant enables long-context serving on B70 where BF16
either OOMs at init or stalls at low concurrency. BF16 is included as a canary.

Context lengths: **16384** and **32768**. ISL/OSL are scaled with context
(ISL = ctx/4, OSL = ctx/8). Concurrency sweep: `{1,2,4,8,16,32,64}`.
TP parallelism same as Test 2.

```bash
# All three models
bash bench_tq_long_context.sh

# Single model
bash bench_tq_long_context.sh llama31_8b
```

Output:

- `/workspace/bench_results/long_context_<TS>.csv` — max concurrency, throughput,
  TTFT/TPOT, and `OOM` status per (model, config, context_length) combination

---

### Test 4 — Long-context accuracy (RULER suite)

Validates that TurboQuant preserves long-context accuracy at 4K, 16K, and 32K
context using the **RULER** benchmark suite (NIAH variants, CWE, Freq-Word-Extraction, VT, QA)
from `lm-eval-harness`. BF16 is included as a canary (OOM at long context is
surfaced in the results).

Models run **sequentially** because `lm_eval` is CPU-heavy (~30–60 min per eval).
TP=2 BF16 (`qwen25_14b`) gets `gpu_memory_utilization=0.70` to reduce crash
probability; the container is restarted between TP=2 runs to reclaim leaked XPU
memory.

```bash
# All three models
bash bench_tq_long_context_accuracy.sh

# Single model
bash bench_tq_long_context_accuracy.sh llama31_8b

# Override which configs to run (comma-separated)
bash bench_tq_long_context_accuracy.sh --configs bf16 llama31_8b
bash bench_tq_long_context_accuracy.sh --configs turboquant_4bit_nc gemma4_e4b
```

Output:

- `/workspace/bench_results/ruler_<model>_<config>_ctx<N>_<TS>/results.json` — full lm_eval output
- `/workspace/bench_results/ruler_summary_<TS>.csv` — aggregated per (model, config, context_length, task)
- Terminal prints a BF16 vs TQ comparison table with delta scores

---

## Collecting Results

All JSON and CSV outputs are written inside the container at
`/workspace/bench_results/`. Copy them out with:

```bash
docker cp vllm-test:/workspace/bench_results ./bench_results
```
