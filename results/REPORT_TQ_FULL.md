# TurboQuant 4-bit — full 5-category benchmark vs FP8 paper (Intel Arc Pro B70)

TQ-only measured (`turboquant_4bit_nc`, fp8 weights; Gemma-4 bf16 weights). BF16/FP8 columns are the v0.22.1rc1 paper's published values.


## Category 1 — KV-cache capacity (max_model_len=4096)

| Model | TP | BF16 (pub) | FP8 (pub) | TQ-4bit (meas) | TQ/BF16 | TQ/FP8 |
|---|--:|--:|--:|--:|--:|--:|
| Llama-3.1-8B | 1 | 141,824 | 283,584 | 427,840 | 3.02× | 1.51× |
| DeepSeek-R1-7B | 1 | 328,576 | 656,000 | 911,680 | 2.77× | 1.39× |
| Gemma-3-1B | 1 | 906,975 | 1,813,950 | 2,615,296 | 2.88× | 1.44× |
| Qwen3-8B | 1 | 123,584 | 247,232 | 381,696 | 3.09× | 1.54× |
| Gemma-4-E4B | 2 | 652,211 | 1,304,423 | 1,581,376 | 2.42× | 1.21× |
| Qwen2.5-14B | 2 | 198,720 | 397,504 | 626,368 | 3.15× | 1.58× |
| Mistral-Small-24B | 2 | 177,216 | 354,432 | 589,312 | 3.33× | 1.66× |
| DeepSeek-R1-70B | 4 | 112,000 | 224,000 | 428,096 | 3.82× | 1.91× |
| Llama-3.3-70B | 4 | 110,336 | 224,000 | 426,816 | 3.87× | 1.91× |
| Qwen2.5-72B | 4 | 102,016 | 204,032 | 383,168 | 3.76× | 1.88× |

## Category 2 — Output throughput (tok/s), 6 scenarios

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

## Category 3 — SLA-bounded max concurrency (p99 TTFT≤5s, TPOT≤200ms)

| Model | TP | max concurrency (TQ) | weights |
|---|--:|--:|:--|
| Llama-3.1-8B | 1 | 32 | fp8 |
| DeepSeek-R1-7B | 1 | 32 | fp8 |
| Gemma-3-1B | 1 | 128 | fp8 |
| Qwen3-8B | 1 | 32 | fp8 |
| Gemma-4-E4B | 2 | 32 | bf16_fallback |
| Qwen2.5-14B | 2 | 16 | fp8 |
| Mistral-Small-24B | 2 | 16 | fp8 |
| DeepSeek-R1-70B | 4 | 8 | fp8 |
| Llama-3.3-70B | 4 | 8 | fp8 |
| Qwen2.5-72B | 4 | 8 | fp8 |

## Category 4 — Long-context peak output throughput (tok/s)

| Model | TP | peak@16K | peak@32K |
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

## Category 5 — RULER accuracy (composite, TQ-4bit)

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

_'—' = skipped (ISL+OSL>maxlen), OOM, or not yet run. Cat-5 composite = unweighted mean of ruler sub-task scores in the CSV._

