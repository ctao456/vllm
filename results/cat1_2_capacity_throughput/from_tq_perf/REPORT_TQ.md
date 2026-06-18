# TurboQuant (4bit_nc) on Intel Arc Pro B70 — measured vs FP8 paper

Config: v0.22.1rc1 paper settings (max_model_len=4096, gpu-util=0.90, fp8 weights, block-size 64, enforce-eager). TQ KV cache = `turboquant_4bit_nc`. FP8/BF16 columns are the paper's **published** capacity; TQ tokens are **measured** here.


## 1. KV-cache capacity (max_model_len=4096)

| Model | TP | BF16 (pub) | FP8 (pub) | TQ-4bit (meas) | TQ vs BF16 | TQ vs FP8 | weights |
|---|--:|--:|--:|--:|--:|--:|:--|
| Llama-3.1-8B | 1 | 141,824 | 283,584 | 427,840 | 3.02× | 1.51× | fp8 |
| DeepSeek-R1-7B | 1 | 328,576 | 656,000 | 911,680 | 2.77× | 1.39× | fp8 |
| Gemma-3-1B | 1 | 906,975 | 1,813,950 | 2,615,296 | 2.88× | 1.44× | fp8 |
| Qwen3-8B | 1 | 123,584 | 247,232 | 381,696 | 3.09× | 1.54× | fp8 |
| Gemma-4-E4B | 2 | 652,211 | 1,304,423 | 1,581,376 | 2.42× | 1.21× | bf16_fallback |
| Qwen2.5-14B | 2 | 198,720 | 397,504 | 626,368 | 3.15× | 1.58× | fp8 |
| Mistral-Small-24B | 2 | 177,216 | 354,432 | 589,312 | 3.33× | 1.66× | fp8 |
| DeepSeek-R1-70B | 4 | 112,000 | 224,000 | 428,096 | 3.82× | 1.91× | fp8 |
| Llama-3.3-70B | 4 | 110,336 | 224,000 | 426,816 | 3.87× | 1.91× | fp8 |
| Qwen2.5-72B | 4 | 102,016 | 204,032 | 383,168 | 3.76× | 1.88× | fp8 |

## 2. Output throughput (tok/s), TQ-4bit — measured

| Model | TP | short_decode | decode_heavy | mixed | high_load | long_prefill | very_long_prefill |
|---|--:|--:|--:|--:|--:|--:|--:|
| Llama-3.1-8B | 1 | 992 | 965 | 799 | 803 | — | — |
| DeepSeek-R1-7B | 1 | 1120 | 1102 | 902 | 895 | — | — |
| Gemma-3-1B | 1 | 769 | 793 | 778 | 1512 | — | — |
| Qwen3-8B | 1 | 912 | 886 | 744 | 786 | — | — |
| Gemma-4-E4B | 2 | 516 | 530 | 513 | 841 | — | — |
| Qwen2.5-14B | 2 | 584 | 601 | 548 | 715 | — | — |
| Mistral-Small-24B | 2 | 729 | 747 | 647 | 602 | — | — |
| DeepSeek-R1-70B | 4 | 355 | 361 | 314 | 314 | — | — |
| Llama-3.3-70B | 4 | 323 | 359 | 304 | 314 | — | — |
| Qwen2.5-72B | 4 | 355 | 360 | 309 | 302 | — | — |

_'—' in throughput = scenario skipped (ISL+OSL>4096) or not yet run._
_Gemma-4 ran bf16 weights (fp8 weight-quant unsupported by XPU fp8 GEMM kernel for its MoE architecture); all others fp8 weights._

