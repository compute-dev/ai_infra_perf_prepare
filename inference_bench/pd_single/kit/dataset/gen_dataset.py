#!/usr/bin/env python3
"""
Dataset generator for SLO-constrained QPM evaluation
====================================================

Purpose: produce evaluation datasets for vLLM `vllm bench serve` whose lengths are exactly controlled and
      may vary per request, so the maximum QPM under a fixed SLO can be measured.

Rationale:
  - What is measured is throughput (QPM) under an SLO, which depends only on the input/output token length
    distribution and the prefix-cache hit rate, not on the semantic content of the prompt.
  - This script therefore pads meaningless placeholder tokens up to an exact target token length and writes
    the vLLM sharegpt format (which can keep a different output length per request).
  - Two length distributions are supported:
      * uniform   — reproduces the symmetric uniform sampling of vLLM `--random-range-ratio` (suits 16k/32k)
      * lognormal — a heavy-tailed distribution anchored on a mean plus a coefficient of variation (CV),
                    which suits the prefill-bound 256k scenario

Correctness guarantees (aligned with vLLM RandomDataset, to avoid length drift):
  1. special-token compensation: real_len = target_len - tokenizer.num_special_tokens_to_add()
  2. three-step token alignment: decode -> re-encode and truncate -> decode
     (the string built from N consecutive tokens does not necessarily re-tokenize to N, so it must be truncated)

Output format (sharegpt, natively supported by vLLM):
  [
    {"conversations": [
        {"from": "human", "value": "<placeholder text that tokenizes to exactly input_len>"},
        {"from": "gpt",   "value": "<placeholder text that tokenizes to exactly output_len>"}
    ]},
    ...
  ]

Usage examples are at the end of the file under __main__, or in the README.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

try:
    from transformers import AutoTokenizer
except ImportError:
    print("transformers is required:  pip install transformers", file=sys.stderr)
    raise


# --------------------------------------------------------------------------- #
# scenario presets
# --------------------------------------------------------------------------- #
# The length parameters per scenario (mean / coefficient of variation / right-tail truncation factor) are
# synthetic parameters fixed by the evaluation protocol; generation reads no real data whatsoever (a few
# values were chosen with earlier length statistics in mind, but the dataset contains no real content and no
# real request lengths).
# Truncation = target mean * tail_ratio; after truncating, the mean is renormalized so the arithmetic mean
# still hits the target exactly.
# Note: the 256k dataset of pd_single is produced by gen_prefix_dataset.py, which uses only input_mean /
#   output_mean / out_cv of the 256k scenario — its input length is fixed (shared prefix + fixed-length
#   unique suffix) and only the output length is sampled from the lognormal.
SCENARIOS = {
    "16k": dict(
        input_mean=4500, output_mean=1300,
        dist="lognormal", cv=0.83, out_cv=0.8, tail_ratio=3.4,
        note="16k: input CV~0.83, output CV=0.8, right tail ~3.4x the mean (synthetic parameters fixed by the protocol).",
    ),
    "32k": dict(
        input_mean=24000, output_mean=700,
        dist="lognormal", cv=0.25, out_cv=0.8, tail_ratio=1.35,
        note="32k: the input mean is 75% of the window with a narrow CV=0.25; output CV=0.8 (synthetic parameters fixed by the protocol).",
    ),
    "256k": dict(
        input_mean=85000, output_mean=1200,
        dist="lognormal", cv=0.83, out_cv=0.8, tail_ratio=3.0,
        note="256k: input CV~0.83, output CV=0.8, right tail ~3.0x the mean (synthetic parameters fixed by the protocol; pd_single uses only its means and output CV).",
    ),
}


# --------------------------------------------------------------------------- #
# length sampling
# --------------------------------------------------------------------------- #
def sample_uniform(rng: np.random.Generator, mean: int, range_ratio: float,
                   n: int) -> np.ndarray:
    """Reproduces the symmetric integer uniform sampling of vLLM RandomDataset:
    the closed interval [floor(mean*(1-r)), ceil(mean*(1+r))]. Its expectation equals mean."""
    low = math.floor(mean * (1 - range_ratio))
    high = math.ceil(mean * (1 + range_ratio))
    low = max(low, 1)
    return rng.integers(low, high + 1, size=n)


def sample_lognormal(rng: np.random.Generator, mean: float, cv: float,
                     n: int, max_len: int | None = None,
                     min_len: int = 1) -> np.ndarray:
    """Heavy-tailed lognormal sampling anchored on the arithmetic mean and the coefficient of variation cv (=std/mean).
    The underlying normal parameters (mu, sigma) are solved from mean and cv:
        sigma^2 = ln(1 + cv^2)
        mu      = ln(mean) - sigma^2 / 2
    so that the arithmetic expectation of the samples equals mean exactly.

    max_len: the right-tail truncation limit. After truncating, the whole sample is renormalized once so the
             actual arithmetic mean still hits mean exactly.
    """
    if cv <= 0:
        val = int(round(mean))
        if max_len:
            val = min(val, max_len)
        return np.full(n, max(min_len, val), dtype=np.int64)
    sigma2 = math.log(1.0 + cv * cv)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - sigma2 / 2.0
    samples = rng.lognormal(mean=mu, sigma=sigma, size=n)

    if max_len is not None:
        # truncating the right tail lowers the mean; iterative scaling pulls it back to the target (keeping the shape and the truncation)
        cap = float(max_len)
        for _ in range(50):
            clipped = np.clip(samples, min_len, cap)
            cur_mean = clipped.mean()
            if cur_mean <= 0:
                break
            scale = mean / cur_mean
            if abs(scale - 1.0) < 1e-4:
                samples = clipped
                break
            # scale up only the samples below the cap; capped ones stay at the cap (so scaling cannot push them out of range)
            samples = clipped * scale
        samples = np.clip(samples, min_len, cap)
    return np.maximum(min_len, np.round(samples)).astype(np.int64)


# --------------------------------------------------------------------------- #
# exact token alignment (same as vLLM)
# --------------------------------------------------------------------------- #
class TokenPadder:
    """Turns a target token count into text that re-tokenizes to exactly that many tokens."""

    # NOTE: seed_token_text is functional data, not a message. It seeds the padding token pool, so changing
    # it changes every generated dataset and therefore the protocol's dataset fingerprint. Do not "translate" it.
    def __init__(self, tokenizer, seed_token_text: str = "的",
                 compensate_special: bool = True):
        self.tok = tokenizer
        try:
            self.num_special = int(self.tok.num_special_tokens_to_add())
        except Exception:
            self.num_special = 0
        self.compensate_special = compensate_special
        # pick a stable base token pool: ids from a safe range of the vocabulary
        self.vocab_size = self.tok.vocab_size
        special_ids = set(self.tok.all_special_ids or [])
        # take a batch of non-special token ids as the padding source
        pool = [i for i in range(100, min(self.vocab_size, 6000))
                if i not in special_ids]
        if not pool:
            pool = [i for i in range(self.vocab_size) if i not in special_ids]
        self._pool = np.array(pool, dtype=np.int64)

    def make(self, target_len: int, offset: int) -> tuple[str, int]:
        """Returns (text, actual token count). The actual count is made equal to target_len where possible.
        compensate_special: when sharegpt is read, tokenizer(prompt) adds special tokens automatically, so
        they are subtracted up front to make the final prompt_len hit the target."""
        eff = target_len
        if self.compensate_special:
            eff = max(1, target_len - self.num_special)

        # cycle through the pool to produce eff tokens
        idx = (offset + np.arange(eff)) % len(self._pool)
        token_ids = self._pool[idx].tolist()

        # three-step alignment: decode -> re-encode and truncate -> decode
        text = self.tok.decode(token_ids)
        re_enc = self.tok.encode(text, add_special_tokens=False)[:eff]
        # if the re-encode comes out short of eff (rare), top it up once
        if len(re_enc) < eff:
            pad_needed = eff - len(re_enc)
            extra = self._pool[np.arange(pad_needed) % len(self._pool)].tolist()
            re_enc = re_enc + extra
            re_enc = re_enc[:eff]
        text = self.tok.decode(re_enc)
        return text, len(re_enc)


# --------------------------------------------------------------------------- #
# main generation flow
# --------------------------------------------------------------------------- #
@dataclass
class GenStats:
    n: int
    input_mean_target: float
    input_mean_actual: float
    input_p50: int
    input_p90: int
    input_p99: int
    input_max: int
    output_mean_target: float
    output_mean_actual: float
    output_p50: int
    output_p90: int
    output_p99: int


def generate(cfg: dict, tokenizer_path: str, num_prompts: int,
             seed: int, out_path: Path, compensate_special: bool) -> GenStats:
    rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    padder = TokenPadder(tok, compensate_special=compensate_special)

    dist = cfg["dist"]
    if dist == "uniform":
        in_lens = sample_uniform(rng, cfg["input_mean"], cfg["range_ratio"],
                                 num_prompts)
        out_lens = sample_uniform(rng, cfg["output_mean"],
                                  cfg.get("out_range_ratio", cfg["range_ratio"]),
                                  num_prompts)
    elif dist == "lognormal":
        tr = cfg.get("tail_ratio")
        # input truncation: tail_ratio*mean, or the explicit window limit if that is smaller
        in_cap = cfg.get("max_input_len")
        if tr:
            tr_cap = int(round(cfg["input_mean"] * tr))
            in_cap = min(in_cap, tr_cap) if in_cap else tr_cap
        # the output uses its own out_tail_ratio (default 3.4) rather than the input's tail_ratio —
        # otherwise the 32k input tail_ratio=1.35 would flatten the output's heavy tail.
        out_tr = cfg.get("out_tail_ratio", 3.4)
        out_cap = cfg.get("max_output_len")
        if out_tr:
            tr_ocap = int(round(cfg["output_mean"] * out_tr))
            out_cap = min(out_cap, tr_ocap) if out_cap else tr_ocap
        in_lens = sample_lognormal(rng, cfg["input_mean"], cfg["cv"],
                                   num_prompts, max_len=in_cap)
        out_lens = sample_lognormal(rng, cfg["output_mean"],
                                    cfg.get("out_cv", 0.15), num_prompts,
                                    max_len=out_cap)
    else:
        raise ValueError(f"unknown distribution: {dist}")

    offsets = rng.integers(0, max(1, padder.vocab_size), size=num_prompts)

    fmt = cfg.get("format", "sharegpt")
    records = []
    act_in, act_out = [], []
    for i in range(num_prompts):
        p_text, p_len = padder.make(int(in_lens[i]), int(offsets[i]))
        act_in.append(p_len)
        if fmt == "custom":
            # vLLM custom dataset: one {"prompt": ..., "output_tokens": ...} per line, with no length filtering.
            # output_tokens is written per row: with --custom-output-len -1 the load generator reads this field
            # to vary the output length per request; passing --custom-output-len N (not -1) overrides it globally.
            act_out.append(int(out_lens[i]))
            records.append({"prompt": p_text, "output_tokens": int(out_lens[i])})
        else:
            g_text, g_len = padder.make(int(out_lens[i]),
                                        int(offsets[i]) + 7919)  # stagger the offset
            act_out.append(g_len)
            records.append({
                "conversations": [
                    {"from": "human", "value": p_text},
                    {"from": "gpt", "value": g_text},
                ]
            })
        if (i + 1) % 50 == 0 or i + 1 == num_prompts:
            print(f"  progress: {i+1}/{num_prompts}", end="\r", file=sys.stderr)
    print("", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "custom":
        # custom = JSONL, one json object per line
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)

    act_in = np.array(act_in)
    act_out = np.array(act_out)
    return GenStats(
        n=num_prompts,
        input_mean_target=cfg["input_mean"],
        input_mean_actual=float(act_in.mean()),
        input_p50=int(np.percentile(act_in, 50)),
        input_p90=int(np.percentile(act_in, 90)),
        input_p99=int(np.percentile(act_in, 99)),
        input_max=int(act_in.max()),
        output_mean_target=cfg["output_mean"],
        output_mean_actual=float(act_out.mean()),
        output_p50=int(np.percentile(act_out, 50)),
        output_p90=int(np.percentile(act_out, 90)),
        output_p99=int(np.percentile(act_out, 99)),
    )


def main():
    ap = argparse.ArgumentParser(
        description="generate sharegpt datasets with exactly controlled lengths for SLO-constrained QPM evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--tokenizer", required=True,
                    help="path or name of the HF tokenizer (must be the model under test)")
    ap.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                    help="use a preset scenario: 16k / 32k / 256k")
    ap.add_argument("--out", required=True, help="output path (.jsonl for custom, .json for sharegpt)")
    ap.add_argument("--format", choices=["custom", "sharegpt"], default="custom",
                    help="output format. custom (default): JSONL with one {prompt} per line, no length "
                         "filtering by vLLM, and a fixed output length given by --custom-output-len. "
                         "sharegpt: variable output length, but vLLM discards samples with prompt_len>2048, "
                         "which makes it unusable for long datasets.")
    ap.add_argument("--num-prompts", type=int, default=200, help="number of prompts to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-compensate-special", action="store_true",
                    help="do not compensate for special tokens (compensation is on by default so the final prompt_len hits the target)")

# manual overrides (when no preset is used)
    ap.add_argument("--dist", choices=["uniform", "lognormal"])
    ap.add_argument("--input-mean", type=int)
    ap.add_argument("--output-mean", type=int)
    ap.add_argument("--range-ratio", type=float, default=0.3,
                    help="uniform mode: symmetric half-width ratio, [0,1)")
    ap.add_argument("--out-range-ratio", type=float,
                    help="the output ratio in uniform mode (defaults to --range-ratio)")
    ap.add_argument("--cv", type=float, default=None,
                    help="input coefficient of variation std/mean in lognormal mode (the preset is used when omitted)")
    ap.add_argument("--out-cv", type=float, default=None,
                    help="output coefficient of variation in lognormal mode (the preset is used when omitted)")
    ap.add_argument("--max-input-len", type=int, default=None,
                    help="hard context-window limit for the input length (the smaller of this and tail_ratio*mean applies)")
    ap.add_argument("--max-output-len", type=int, default=None,
                    help="hard limit on the output length")
    ap.add_argument("--tail-ratio", type=float, default=None,
                    help="lognormal right-tail truncation = mean * this factor (the preset applies by default)")
    ap.add_argument("--out-tail-ratio", type=float, default=None,
                    help="output right-tail truncation = output mean * this factor (default 3.4; "
                         "it does not reuse --tail-ratio)")
    args = ap.parse_args()

    if args.scenario:
        cfg = dict(SCENARIOS[args.scenario])
        # allow the command line to override the preset distribution/CV/truncation
        if args.dist:
            cfg["dist"] = args.dist
        if args.input_mean:
            cfg["input_mean"] = args.input_mean
        if args.output_mean:
            cfg["output_mean"] = args.output_mean
        if args.cv is not None and cfg["dist"] == "lognormal":
            cfg["cv"] = args.cv
        if args.out_cv is not None and cfg["dist"] == "lognormal":
            cfg["out_cv"] = args.out_cv
        if args.max_input_len is not None:
            cfg["max_input_len"] = args.max_input_len
        if args.max_output_len is not None:
            cfg["max_output_len"] = args.max_output_len
        if args.tail_ratio is not None:
            cfg["tail_ratio"] = args.tail_ratio
        if args.out_tail_ratio is not None:
            cfg["out_tail_ratio"] = args.out_tail_ratio
        cfg["format"] = args.format
        print(f"[scenario {args.scenario}] {cfg['note']}", file=sys.stderr)
    else:
        if not (args.dist and args.input_mean and args.output_mean):
            ap.error("without --scenario you must provide --dist, --input-mean and --output-mean")
        cfg = dict(
            dist=args.dist, input_mean=args.input_mean,
            output_mean=args.output_mean,
            range_ratio=args.range_ratio,
            out_range_ratio=args.out_range_ratio or args.range_ratio,
            cv=args.cv if args.cv is not None else 0.6,
            out_cv=args.out_cv if args.out_cv is not None else 0.15,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            tail_ratio=args.tail_ratio,
            format=args.format,
            note="(manual)",
        )

    print(f"[config] {json.dumps({k: v for k, v in cfg.items() if k != 'note'}, ensure_ascii=False)}",
          file=sys.stderr)

    stats = generate(
        cfg=cfg, tokenizer_path=args.tokenizer, num_prompts=args.num_prompts,
        seed=args.seed, out_path=Path(args.out),
        compensate_special=not args.no_compensate_special,
    )

    print("\n=== generation complete, length verification ===", file=sys.stderr)
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2),
          file=sys.stderr)
    # report how closely the means were hit
    in_err = abs(stats.input_mean_actual - stats.input_mean_target) / stats.input_mean_target
    print(f"\ninput mean: target {stats.input_mean_target} / actual "
          f"{stats.input_mean_actual:.0f}  deviation {in_err*100:.2f}%", file=sys.stderr)
    print(f"output file: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
