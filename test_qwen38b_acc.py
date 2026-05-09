#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""test_qwen35_acc.py — Offline accuracy sanity check for Qwen3-8B.

Qwen3-8B is a pure full-attention model (36 layers, head_dim=128, 32 heads, 8 KV heads).
No sliding window, no hybrid layers — straightforward TQ candidate.

Usage:
  # bf16 baseline
  python test_qwen35_acc.py

  # TurboQuant
  python test_qwen35_acc.py --tq
"""

import argparse
import os

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")

from vllm import LLM, SamplingParams  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tq", action="store_true", help="Use TurboQuant k3v4_nc KV cache"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()

    kv_cache_dtype = "turboquant_k3v4_nc" if args.tq else "auto"

    llm = LLM(
        args.model,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        trust_remote_code=True,
    )

    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=512, temperature=0)

    tests = [
        (
            "capital_france",
            "What is the capital of France? Answer in one word.",
            ["paris"],
        ),
        (
            "capital_japan",
            "What is the capital of Japan? Answer in one word.",
            ["tokyo"],
        ),
        ("math_addition", "What is 137 + 248? Reply with just the number.", ["385"]),
        ("math_multiply", "What is 12 times 15? Reply with just the number.", ["180"]),
        (
            "logic_sequence",
            "What comes next in the sequence 2, 4, 8, 16, ...? Reply with just the number.",
            ["32"],
        ),
        (
            "word_reverse",
            "Reverse the word 'hello'. Reply with just the reversed word.",
            ["olleh"],
        ),
        (
            "translate_es",
            "Translate 'Good morning' to Spanish. Reply with just the translation.",
            ["buenos"],
        ),
        (
            "sentiment",
            "Positive or negative? 'I absolutely love this product!' One word.",
            ["positive"],
        ),
        (
            "coherent_story",
            "Write one sentence about a cat sitting on a windowsill.",
            ["cat", "window"],
        ),
        (
            "code_hello",
            "Write a Python one-liner that prints hello world.",
            ["print", "hello"],
        ),
    ]

    prompts = []
    for _, prompt, _ in tests:
        messages = [{"role": "user", "content": prompt}]
        prompts.append(
            tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    outputs = llm.generate(prompts, sp)

    passed = 0
    label = "TQ-k3v4_nc" if args.tq else "bf16"
    print(f"\n=== Qwen3-8B accuracy check  config={label}")
    for (name, _, keywords), out in zip(tests, outputs):
        text = out.outputs[0].text
        text_lower = text.lower()
        hit = any(kw in text_lower for kw in keywords)
        if hit:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}")
            print(f"        expected: {keywords}")
            print(f"        got:      {text[:200]}")

    total = len(tests)
    print(f"\n=== {passed}/{total} passed  config={label}")
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
