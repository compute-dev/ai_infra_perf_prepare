#!/usr/bin/env python3
# ============================================================================
# torch_peak_probe.py — Single-card GEMM peak (torch side, second-opinion tool)
# ----------------------------------------------------------------------------
# Companion to cublasMatmulBench: measures GEMM peak TFLOPS via torch on the
# real framework path. Input is aligned to cublasMatmulBench's -B=0 (all-zeros).
#
# Single source of truth (2026-09 redesign): matrix dimensions m/n/k are no
# longer hardcoded per card — they are parsed at RUNTIME from the sibling
# cublas_bench_<model>.sh, so torch always uses the exact same shapes as
# cublasMatmulBench (apples-to-apples). Shapes are maintained in one place
# (the cublas script); adding a new card requires zero changes here.
# One script covers H20/H800/H200/H100/B200/B300/GB200/GB300/A800/L20, etc.
#
# GPU model source (priority): --gpu arg > GPU_NAME env var > error + list.
#   (case-insensitive, same convention as auto_cublas_bench.sh)
#
# Precision coverage (determined by -P codes in the cublas script):
#   FP16(hsh) / BF16(tst) / TF32(sss_fast_tf32 -> allow_tf32) / FP32(sss) / FP64(ddd)
#     — torch and cublas are both reliable; should be nearly equal
#   FP8(qqssq -> _scaled_mm e4m3) — torch reliable; cross-checks cublas qqssq
#   INT8(bisb_imma) — torch has no clean int8 peak entry point; this script
#     does NOT test it. INT8 is authoritative from cublas (bisb_imma).
#
# Usage (run inside a container with a torch image):
#   CUDA_VISIBLE_DEVICES=0 GPU_NAME=H800 python3 torch_peak_probe.py       # env var
#   CUDA_VISIBLE_DEVICES=0 python3 torch_peak_probe.py --gpu H20           # arg
#   CUDA_VISIBLE_DEVICES=0 python3 torch_peak_probe.py --gpu H20 --randn   # random input
#   CUDA_VISIBLE_DEVICES=0 python3 torch_peak_probe.py --gpu H20 --iters 100
#   python3 torch_peak_probe.py --gpu H800 --dry-run      # print shapes only (no GPU/torch)
# Output: one line "PREC torch TFLOPS: <val>" per precision (parse-friendly).
# ============================================================================
import argparse, os, re, sys, glob
torch = None  # lazy import: resolve_gpu/load_cases are pure parsing, no torch needed

# cublasMatmulBench -P code -> precision name. bisb_imma (INT8) deliberately
# omitted: torch has no clean int8 peak entry point.
PMAP = {
    "qqssq": "FP8",
    "hsh": "FP16",
    "tst": "BF16",
    "sss_fast_tf32": "TF32",
    "sss": "FP32",
    "ddd": "FP64",
}


def flops(m, n, k, it):  # 2*m*n*k per matmul
    return 2.0 * m * n * k * it


def bench(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 1000.0  # sec


def mk(m, k, dtype, zero, dev):
    if zero:
        return torch.zeros((m, k), dtype=dtype, device=dev)
    return torch.randn((m, k), dtype=dtype, device=dev)


def resolve_gpu(cli_gpu, bench_dir):
    """Determine GPU model (upper case). Priority: --gpu, then GPU_NAME env var."""
    gpu = cli_gpu or os.environ.get("GPU_NAME")
    avail = sorted(
        name
        for p in glob.glob(os.path.join(bench_dir, "cublas_bench_*.sh"))
        for name in [os.path.basename(p)[len("cublas_bench_"):-len(".sh")]]
        # only list standard models (all-caps alphanumeric); filter out underlying binary aliases
        if re.fullmatch(r"[A-Z0-9]+", name)
    )
    if not gpu:
        sys.exit(
            "Error: GPU model not specified. Use --gpu <model> or export GPU_NAME=<model>.\n"
            f"Available models in this directory: {' '.join(avail) if avail else '(no cublas_bench_*.sh found)'}"
        )
    gpu = gpu.upper()
    if not os.path.isfile(os.path.join(bench_dir, f"cublas_bench_{gpu}.sh")):
        sys.exit(
            f"Error: cublas_bench_{gpu}.sh not found (in {bench_dir}).\n"
            f"Available models: {' '.join(avail) if avail else '(none)'}"
        )
    return gpu


def load_cases(gpu, bench_dir):
    """Parse cublas_bench_<model>.sh, extract -P/-m/-n/-k per line, build [(name, m, n, k, kind), ...]."""
    path = os.path.join(bench_dir, f"cublas_bench_{gpu}.sh")
    cases = []
    seen = set()
    with open(path) as fh:
        for line in fh:
            mP = re.search(r"-P=(\w+)", line)
            mm = re.search(r"-m=(\d+)", line)
            mn = re.search(r"-n=(\d+)", line)
            mk_ = re.search(r"-k=(\d+)", line)
            if not (mP and mm and mn and mk_):
                continue
            code = mP.group(1)
            if code not in PMAP:  # INT8 (bisb_imma) etc. land here and are skipped
                continue
            name = PMAP[code]
            if name in seen:  # one shape per precision (prevent duplicates in script)
                continue
            seen.add(name)
            cases.append((name, int(mm.group(1)), int(mn.group(1)), int(mk_.group(1)), name.lower()))
    return cases, path


def build_fn(kind, m, n, k, zero, dev):
    """Return a callable GEMM closure for the given precision."""
    if kind in ("fp16", "bf16"):
        dt = torch.float16 if kind == "fp16" else torch.bfloat16
        A = mk(m, k, dt, zero, dev); B = mk(k, n, dt, zero, dev)
        return lambda: torch.matmul(A, B)
    if kind == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        A = mk(m, k, torch.float32, zero, dev); B = mk(k, n, torch.float32, zero, dev)
        return lambda: torch.matmul(A, B)
    if kind == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        A = mk(m, k, torch.float32, zero, dev); B = mk(k, n, torch.float32, zero, dev)
        return lambda: torch.matmul(A, B)
    if kind == "fp64":
        A = mk(m, k, torch.float64, zero, dev); B = mk(k, n, torch.float64, zero, dev)
        return lambda: torch.matmul(A, B)
    if kind == "fp8":
        # torch._scaled_mm: A row-major (m,k) e4m3, B column-major (k,n) e4m3
        e4m3 = torch.float8_e4m3fn
        A = mk(m, k, torch.float16, zero, dev).to(e4m3)
        B = mk(n, k, torch.float16, zero, dev).to(e4m3).t()  # (k,n) col-major
        sa = torch.tensor(1.0, device=dev); sb = torch.tensor(1.0, device=dev)
        return lambda: torch._scaled_mm(A, B, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None, help="GPU model (e.g. H20/H800); default: env GPU_NAME")
    ap.add_argument("--bench-dir", default=None, help="Directory containing cublas_bench_*.sh (default: script dir)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--randn", action="store_true", help="Random input (default: all-zeros, aligned with cublas -B=0)")
    ap.add_argument("--dry-run", action="store_true", help="Only parse and print shapes; do not run GEMM (no GPU/torch needed)")
    a = ap.parse_args()

    bench_dir = a.bench_dir or os.path.dirname(os.path.abspath(__file__))
    gpu = resolve_gpu(a.gpu, bench_dir)
    cases, src = load_cases(gpu, bench_dir)

    zero = not a.randn
    it = a.iters

    # Diagnostic info to stderr; stdout only has "PREC torch TFLOPS:" lines (parse-friendly)
    print(f"[torch_peak_probe] GPU={gpu}  shape source={os.path.basename(src)}  "
          f"input={'zero (aligned with cublas -B=0)' if zero else 'randn'}  iters={it}", file=sys.stderr)
    print(f"[torch_peak_probe] {len(cases)} precision(s) parsed: "
          f"{', '.join(c[0] for c in cases)}  (INT8 is cublas-only, not tested here)", file=sys.stderr)

    # Only at this point is torch actually needed (selection/parsing don't depend on it,
    # allowing shape verification on a GPU-less controller)
    if a.dry_run:
        for name, m, n, k, _ in cases:
            print(f"{name} m={m} n={n} k={k}", file=sys.stderr)
        return
    global torch
    import torch  # noqa: E402
    dev = "cuda"
    assert torch.cuda.is_available(), "no CUDA"

    for name, m, n, k, kind in cases:
        try:
            fn = build_fn(kind, m, n, k, zero, dev)
            if fn is None:
                continue
            t = bench(fn, it)
            tflops = flops(m, n, k, it) / t / 1e12
            print(f"{name} torch TFLOPS: {tflops:.1f}")
        except Exception as ex:
            print(f"{name} torch TFLOPS: NA  ({type(ex).__name__}: {str(ex)[:80]})")


if __name__ == "__main__":
    main()
