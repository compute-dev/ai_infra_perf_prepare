#!/usr/bin/env python3
"""
Generator for datasets with a controllable prefix-cache hit rate
===============================================================

Purpose: measure what prefix caching buys for TTFT. Every request shares one common prefix and then
      appends a suffix that is unique to it, which fixes the prefix-cache hit rate at:

        hit_rate ~= shared_tokens / (shared_tokens + unique_tokens)

      Note that a prefix cache only helps for prefixes that are token-aligned with an already cached
      sequence. The shared prefix must therefore be token-for-token identical (its offset must not vary
      per row), so it is generated once from a fixed token sequence and only the suffix is per-row unique.

      The measured hit rate is also affected by chunked prefill block boundaries: if the suffix does not
      start on a 16-token boundary, the first block may count as a miss. shared_tokens is therefore kept a
      multiple of 16 to align as well as possible. The authoritative hit rate is the one from the server's
      /metrics: vllm:prefix_cache_hits_total / prefix_cache_queries_total.

Output format: same as --format custom of gen_dataset.py
    JSONL, one object per line: {"prompt": <shared prefix + unique suffix>, "output_tokens": <variable output>}
    Use it with --skip-chat-template and --custom-output-len -1 of vllm bench serve.

Usage:
  python3 gen_prefix_dataset.py \
      --tokenizer <model dir> \
      --scenario 256k \
      --prefix-hit-ratio 0.85 \
      --num-prompts 60 --seed 42 \
      --out datasets/dataset_256k_prefix85.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

# reuse the length-distribution scenarios of gen_dataset.py (input mean / output mean / CV / truncation)
from gen_dataset import SCENARIOS, sample_lognormal, TokenPadder


def build_prefix(padder: TokenPadder, n_tokens: int) -> tuple[str, int]:
    """Builds a fixed shared-prefix text that is token-for-token identical and exactly n_tokens long.
    The offset is pinned to 0 so that every request has exactly the same prefix token sequence."""
    # align n_tokens to 16 (the prefix-cache block boundary) to reduce boundary misses
    n = (n_tokens // 16) * 16
    if n < 16:
        n = 16
    text, real = padder.make(n, offset=0)
    return text, real


def build_suffix(padder: TokenPadder, n_tokens: int, offset: int) -> tuple[str, int]:
    """The per-request unique suffix; offsets are staggered so no two suffixes are the same."""
    return padder.make(max(1, n_tokens), offset=offset)


@dataclass
class Stats:
    n: int
    prefix_tokens: int
    suffix_mean: int
    suffix_min: int
    suffix_max: int
    total_mean: int
    hit_ratio_target: float
    hit_ratio_actual: float
    output_mean_actual: float


def main():
    ap = argparse.ArgumentParser(
        description="generate a custom JSONL dataset with a controllable prefix-cache hit rate")
    ap.add_argument("--tokenizer", required=True, help="path to the HF tokenizer (must be the model under test)")
    ap.add_argument("--scenario", choices=list(SCENARIOS.keys()), required=True)
    ap.add_argument("--prefix-hit-ratio", type=float, default=0.85,
                    help="target prefix hit rate = shared prefix / (shared prefix + unique suffix), default 0.85")
    ap.add_argument("--num-prompts", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offset-base", type=int, default=1,
                    help="deterministic offset base for the suffixes. Using a different value per file keeps "
                         "suffixes globally unique, which avoids hit-rate drift from resampling across tiers.")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    args = ap.parse_args()

    cfg = dict(SCENARIOS[args.scenario])
    input_mean = cfg["input_mean"]
    rng = np.random.default_rng(args.seed)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    padder = TokenPadder(tok, compensate_special=True)

    # target hit rate -> the token split between shared prefix and unique suffix
    ratio = min(max(args.prefix_hit_ratio, 0.0), 0.99)
    shared_target = int(round(input_mean * ratio))
    unique_target = max(1, int(round(input_mean - shared_target)))

    # the shared prefix is generated once (a fixed sequence)
    prefix_text, prefix_real = build_prefix(padder, shared_target)

    # unique suffix: per request, input_mean minus the shared prefix, so the total stays ~= input_mean
    # (a fixed-length suffix is cleaner; the output length varies)
    out_cfg_cv = cfg.get("out_cv", 0.8)
    out_tr = cfg.get("out_tail_ratio", 3.4)
    out_cap = int(round(cfg["output_mean"] * out_tr))
    out_lens = sample_lognormal(rng, cfg["output_mean"], out_cfg_cv,
                                args.num_prompts, max_len=out_cap)

    # the unique suffix uses a deterministic offset: offset_base + i + 1.
    # Use a different --offset-base per file to keep every suffix globally unique (the shared prefix must be
    # token-for-token identical for the hit rate to hold steady).
    records = []
    total_lens = []
    suffix_lens = []
    offset_base = args.offset_base
    for i in range(args.num_prompts):
        suffix_text, suffix_real = build_suffix(
            padder, unique_target, offset_base + i + 1)
        prompt = prefix_text + suffix_text
        # the concatenated token count is taken as measured prefix + measured suffix (skipping an expensive
        # re-encode of the whole text; the hit rate comes from the server's /metrics anyway, and a 1-2 token
        # merge error at the boundary is negligible)
        full_len = prefix_real + suffix_real
        total_lens.append(full_len)
        suffix_lens.append(suffix_real)
        records.append({
            "prompt": prompt,
            "output_tokens": int(out_lens[i]),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_mean = float(np.mean(total_lens))
    stats = Stats(
        n=args.num_prompts,
        prefix_tokens=prefix_real,
        suffix_mean=float(np.mean(suffix_lens)),
        suffix_min=int(np.min(suffix_lens)),
        suffix_max=int(np.max(suffix_lens)),
        total_mean=total_mean,
        hit_ratio_target=ratio,
        hit_ratio_actual=prefix_real / total_mean if total_mean else 0.0,
        output_mean_actual=float(np.mean(out_lens)),
    )
    print("\n=== generation complete ===", file=sys.stderr)
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2), file=sys.stderr)
    print(f"output file: {args.out}", file=sys.stderr)
    print("note: the authoritative hit rate is the server's "
          "vllm:prefix_cache_hits_total / prefix_cache_queries_total from /metrics.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
