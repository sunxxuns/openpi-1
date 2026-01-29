#!/usr/bin/env python3
"""
Microbench: compare backend kernel choices for the hot small-BMM shapes in policy inference.

Hot shapes seen in the policy replay trace:
  - bmm1: [B=8, M=11, K=799] @ [B=8, K=799, N=256] -> [8, 11, 256]
  - bmm2: [B=8, M=11, K=256] @ [B=8, K=256, N=799] -> [8, 11, 799]

We benchmark and (optionally) dump a Chrome trace, then print the top kernel names.

Usage examples:
  python scripts/bench_small_bmm_kernels.py
  TORCH_ROCM_USE_HIPBLASLT=0 python scripts/bench_small_bmm_kernels.py
  TORCH_ROCM_USE_HIPBLASLT=1 python scripts/bench_small_bmm_kernels.py

  DUMP_TRACE=1 python scripts/bench_small_bmm_kernels.py
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile


def _top_kernels_from_trace(path: str, *, prefix: str = "Cijk_", top_k: int = 15) -> list[tuple[str, float, int]]:
    with open(path, "r") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    dur = defaultdict(float)
    cnt = defaultdict(int)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name")
        if not isinstance(name, str) or not name.startswith(prefix):
            continue
        d = ev.get("dur")
        if d is None:
            continue
        dur[name] += float(d)
        cnt[name] += 1
    items = sorted(dur.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(n, d_us / 1000.0, cnt[n]) for n, d_us in items]  # ms


def main() -> None:
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print(f"ROCm:    {getattr(torch.version, 'hip', None)}")
    print(f"Env TORCH_ROCM_USE_HIPBLASLT={os.environ.get('TORCH_ROCM_USE_HIPBLASLT')}")

    B, M, K, N = 8, 11, 799, 256
    a1 = torch.randn(B, M, K, device=device, dtype=dtype)
    b1 = torch.randn(B, K, N, device=device, dtype=dtype)

    a2 = torch.randn(B, M, N, device=device, dtype=dtype)
    b2 = torch.randn(B, N, K, device=device, dtype=dtype)

    def bench(fn, warmup: int = 200, iters: int = 2000) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / iters

    bmm1 = lambda: torch.bmm(a1, b1)
    bmm2 = lambda: torch.bmm(a2, b2)
    # Alternative entrypoints (can trigger different internal GEMM paths)
    baddbmm1 = lambda: torch.baddbmm(torch.zeros(B, M, N, device=device, dtype=dtype), a1, b1)
    baddbmm2 = lambda: torch.baddbmm(torch.zeros(B, M, K, device=device, dtype=dtype), a2, b2)

    t1 = bench(bmm1)
    t2 = bench(bmm2)
    t1b = bench(baddbmm1, warmup=100, iters=1000)
    t2b = bench(baddbmm2, warmup=100, iters=1000)
    print(f"\nBMM1 [8,11,799]@[8,799,256]  : {t1:.4f} ms/iter")
    print(f"BMM2 [8,11,256]@[8,256,799]  : {t2:.4f} ms/iter")
    print(f"BADD BMM1 (beta=1)           : {t1b:.4f} ms/iter")
    print(f"BADD BMM2 (beta=1)           : {t2b:.4f} ms/iter")

    if os.environ.get("DUMP_TRACE", "0") == "1":
        out = os.environ.get("TRACE_OUT", "traces/small_bmm_trace.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        print(f"\nProfiling and exporting trace to {out} ...")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_flops=False,
        ) as prof:
            # Keep it short but representative
            for _ in range(100):
                bmm1()
                bmm2()
            torch.cuda.synchronize()
        prof.export_chrome_trace(out)
        print("Top Cijk_* kernels in trace:")
        for name, ms, calls in _top_kernels_from_trace(out):
            print(f"  {ms:.3f} ms  calls={calls:4d}  {name[:140]}")


if __name__ == "__main__":
    main()

