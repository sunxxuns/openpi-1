#!/usr/bin/env python3
"""Quick GEMM tuner for OpenPI M=788 and M=11 shapes.

Uses known-good hipBLASLt solver indices from the M=532 tuned configs (same N,K)
and benchmarks them against rocBLAS (torch.mm) for the new M values.
"""

import csv
import os
import sys
import time

import torch

os.environ.setdefault("AMD_LOG_LEVEL", "0")
os.environ.setdefault("HIP_LAUNCH_BLOCKING", "0")

GPU_ID = int(os.environ.get("OPENPI_GPU_ID", "7"))
DEVICE = torch.device(f"cuda:{GPU_ID}")
torch.cuda.set_device(DEVICE)

from aiter import hipb_create_extension, hipb_mm  # noqa: E402
from aiter.jit.utils.chip_info import get_cu_num  # noqa: E402

try:
    from aiter import gemm_a16w16_asm
    HAS_ASM = True
except Exception:
    HAS_ASM = False

hipb_create_extension()

WARMUP = 30
ITERS = 200
CU_NUM = get_cu_num()

# Known-good solver indices from M=532 tuned configs
# Format: (N, K) -> list of (libtype, solidx, splitK) to try
KNOWN_GOOD = {
    (2048, 2048):   [("hipblaslt", 618302, 0), ("hipblaslt", 618372, 0), ("hipblaslt", 618334, 0)],
    (2048, 16384):  [("hipblaslt", 618302, 0)],
    (2560, 2048):   [("hipblaslt", 618372, 0), ("hipblaslt", 618302, 0)],
    (32768, 2048):  [("hipblaslt", 618334, 0), ("hipblaslt", 618302, 0)],
    # For M=11 (expert) shapes - try same + some common gfx950 indices
    (1024, 2048):   [("hipblaslt", 618302, 0), ("hipblaslt", 618372, 0), ("hipblaslt", 618373, 0)],
    (2560, 1024):   [("hipblaslt", 618372, 0), ("hipblaslt", 618302, 0), ("hipblaslt", 618373, 0)],
    (8192, 1024):   [("hipblaslt", 618302, 0), ("hipblaslt", 618334, 0)],
    (1024, 4096):   [("hipblaslt", 618302, 0), ("hipblaslt", 618372, 0)],
}

# Shapes to tune: (M, N, K, bias)
SHAPES = [
    # Gemma main model prefill (M=788, 18 layers × 1 pass)
    (788, 2048, 2048, False),
    (788, 2560, 2048, False),
    (788, 32768, 2048, False),
    (788, 2048, 16384, False),
    # Gemma expert denoising (M=11, 10 layers × 10 steps = 100 passes)
    (11, 1024, 2048, False),
    (11, 2560, 1024, False),
    (11, 8192, 1024, False),
    (11, 1024, 4096, False),
]


def bench_hipblaslt(M, N, K, bias, solidx):
    """Benchmark a single hipBLASLt solver index. Returns us per call."""
    a = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
    b = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)
    bias_t = torch.randn(N, dtype=torch.bfloat16, device=DEVICE) if bias else None

    # Verify it works
    try:
        out = hipb_mm(a, b, solidx, bias=bias_t, out_dtype=torch.bfloat16)
        if out is None or out.shape != (M, N):
            return None
    except Exception:
        return None

    # Warmup
    for _ in range(WARMUP):
        hipb_mm(a, b, solidx, bias=bias_t, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        hipb_mm(a, b, solidx, bias=bias_t, out_dtype=torch.bfloat16)
    end.record()
    end.synchronize()
    us = start.elapsed_time(end) * 1000.0 / ITERS
    return us


def bench_torch_mm(M, N, K, bias):
    """Benchmark torch.mm (rocBLAS). Returns us per call."""
    a = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
    w = torch.randn(K, N, dtype=torch.bfloat16, device=DEVICE)
    bias_t = torch.randn(N, dtype=torch.bfloat16, device=DEVICE) if bias else None

    def run():
        out = torch.mm(a, w)
        if bias_t is not None:
            out = out + bias_t
        return out

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / ITERS


def bench_asm(M, N, K):
    """Benchmark ASM GEMM kernels. Returns (us, solidx, splitK) or (None, None, None)."""
    if not HAS_ASM or N % 64 != 0 or K % 64 != 0:
        return None, None, None

    a = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
    b = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)

    best_us, best_solidx, best_splitk = float('inf'), -1, 1

    for splitk in [1, 2, 4, 8]:
        for solidx in range(16):
            try:
                out = gemm_a16w16_asm(a, b, solidx, splitk)
                if out is None or out.shape != (M, N):
                    continue
                for _ in range(10):
                    gemm_a16w16_asm(a, b, solidx, splitk)
                torch.cuda.synchronize()

                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(ITERS):
                    gemm_a16w16_asm(a, b, solidx, splitk)
                e.record()
                e.synchronize()
                us = s.elapsed_time(e) * 1000.0 / ITERS
                if us < best_us:
                    best_us, best_solidx, best_splitk = us, solidx, splitk
            except Exception:
                continue

    return (best_us, best_solidx, best_splitk) if best_solidx >= 0 else (None, None, None)


def tune_shape(M, N, K, bias):
    print(f"\n{'='*60}")
    print(f"  M={M}, N={N}, K={K}, bias={bias}")
    print(f"{'='*60}")

    # Baseline: torch.mm (rocBLAS)
    torch_us = bench_torch_mm(M, N, K, bias)
    print(f"  torch.mm (rocBLAS):  {torch_us:8.1f} us")

    # Test known-good hipBLASLt solutions
    best_hb_us, best_hb_solidx = float('inf'), -1
    candidates = KNOWN_GOOD.get((N, K), [])
    for libtype, solidx, splitk in candidates:
        if libtype == "hipblaslt":
            us = bench_hipblaslt(M, N, K, bias, solidx)
            if us is not None:
                tag = " <-- best" if us < best_hb_us else ""
                print(f"  hipBLASLt [{solidx:>6}]:  {us:8.1f} us{tag}")
                if us < best_hb_us:
                    best_hb_us, best_hb_solidx = us, solidx

    # Test ASM (only no-bias)
    asm_us, asm_solidx, asm_splitk = None, None, None
    if not bias:
        asm_us, asm_solidx, asm_splitk = bench_asm(M, N, K)
        if asm_us is not None:
            print(f"  ASM [sol={asm_solidx},sk={asm_splitk}]:  {asm_us:8.1f} us")

    # Pick winner
    options = [("torch", torch_us, 0, 0, "")]
    if best_hb_solidx >= 0:
        options.append(("hipblaslt", best_hb_us, best_hb_solidx, 0, ""))
    if asm_us is not None:
        options.append(("asm", asm_us, asm_solidx, asm_splitk, ""))

    winner = min(options, key=lambda x: x[1])
    libtype, us, solidx, splitk, kname = winner

    flops = 2.0 * M * N * K
    tflops = flops / (us * 1e-6) / 1e12
    bw_bytes = (M * K + N * K + M * N) * 2
    bw_gbs = bw_bytes / (us * 1e-6) / 1e9
    speedup = torch_us / us if us > 0 else 1.0

    print(f"\n  >>> WINNER: {libtype} solidx={solidx} splitK={splitk}")
    print(f"      {us:.1f} us | {tflops:.2f} TFLOPS | {speedup:.2f}x vs rocBLAS")

    return {
        "cu_num": CU_NUM, "M": M, "N": N, "K": K,
        "bias": bias, "dtype": "torch.bfloat16", "outdtype": "torch.bfloat16",
        "scaleAB": False, "bpreshuffle": False,
        "libtype": libtype, "solidx": solidx, "splitK": splitk,
        "us": round(us, 4), "kernelName": kname,
        "err_ratio": 0.0, "tflops": round(tflops, 2), "bw": round(bw_gbs, 2),
    }


def main():
    print(f"OpenPI GEMM Tuner (M=788/M=11 shapes)")
    print(f"Device: cuda:{GPU_ID} | CU: {CU_NUM}")

    results = []
    for M, N, K, bias in SHAPES:
        r = tune_shape(M, N, K, bias)
        if r["libtype"] != "torch":
            results.append(r)

    # Read existing CSV, merge
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "openpi_bf16_tuned_gemm.csv"))

    existing = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = list(csv.DictReader(f))

    # Keep configs for M values we didn't tune
    new_ms = {str(r["M"]) for r in results}
    merged = [r for r in existing if r.get("M") not in new_ms]
    fnames = ["cu_num","M","N","K","bias","dtype","outdtype","scaleAB","bpreshuffle",
              "libtype","solidx","splitK","us","kernelName","err_ratio","tflops","bw"]
    for r in results:
        merged.append({k: r[k] for k in fnames})

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fnames)
        w.writeheader()
        for row in merged:
            w.writerow(row)

    print(f"\n{'='*60}")
    print(f"Wrote {len(results)} new + {len(merged)-len(results)} existing = {len(merged)} total to {out_path}")

    # Remove cached /tmp config
    tmp_cfg = "/tmp/aiter_configs/bf16_tuned_gemm.csv"
    if os.path.exists(tmp_cfg):
        os.remove(tmp_cfg)
        print(f"Cleared cache: {tmp_cfg}")


if __name__ == "__main__":
    main()
