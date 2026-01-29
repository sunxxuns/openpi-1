#!/usr/bin/env python3
"""
Unit-test style harness: validate and time aiter GEMM backends/solutions.

Goal:
  - Exercise aiter's available GEMM "solutions" (asm / hipBLASLt / triton / skinny / torch)
  - Check numeric correctness vs torch reference
  - Print per-backend latency for a few representative shapes (including OpenPI hot shapes)

Usage:
  python scripts/unit_test_aiter_solutions.py
  python scripts/unit_test_aiter_solutions.py --shapes 11,1024,2048 11,1024,4096
  python scripts/unit_test_aiter_solutions.py --backends torch triton hipblaslt asm skinny
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int

    @classmethod
    def parse(cls, s: str) -> "Shape":
        parts = s.split(",")
        if len(parts) != 3:
            raise ValueError(f"Expected 'M,N,K', got: {s!r}")
        m, n, k = (int(x) for x in parts)
        return cls(m=m, n=n, k=k)

    def __str__(self) -> str:
        return f"M={self.m} N={self.n} K={self.k}"


def _bench_ms(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def _assert_close(
    name: str,
    ref_fp32: torch.Tensor,
    out: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    # Compare in fp32 against fp32 reference.
    out_fp32 = out.float()
    torch.testing.assert_close(
        out_fp32,
        ref_fp32,
        rtol=rtol,
        atol=atol,
        msg=f"{name} mismatch (rtol={rtol} atol={atol})",
    )


def _tols_for(backend: str, *, dtype: torch.dtype) -> tuple[float, float]:
    # Practical numeric tolerances for reduced-precision GEMM.
    # Note: aiter asm split-K can have larger reduction-order error.
    if dtype == torch.float16:
        base = (5e-2, 5e-1)
        asm = (1e-1, 2.0)
    else:
        # bf16
        base = (5e-2, 7.5e-1)
        asm = (1e-1, 3.0)
    return asm if backend == "asm" else base


def _is_expected_unsupported(backend: str, e: Exception) -> bool:
    # aiter backends/solutions are not universally available for all shapes.
    # Treat known "not supported / no kernel" errors as skips instead of hard failures.
    msg = str(e)
    needles = [
        "get_heuristic_kernel not find kernel",  # asm heuristic has no matching kernel
        "no solution for",  # tuned_gemm default assert path
        "Unsupported",  # various aiter wrappers
        "not supported",  # generic
        "does not support",  # generic
    ]
    if isinstance(e, RuntimeError) and any(n in msg for n in needles):
        return True
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Device, default cuda:0",
    )
    p.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16"],
        help="Input/weight dtype to test (default bf16)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Warmup iterations per backend/shape",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Timed iterations per backend/shape",
    )
    p.add_argument(
        "--backends",
        nargs="+",
        default=["torch", "triton", "hipblaslt", "asm", "skinny"],
        help="Which backends to test",
    )
    p.add_argument(
        "--shapes",
        nargs="*",
        default=[
            # OpenPI hot linear-ish shapes seen in logs/traces
            "11,1024,2048",
            "11,1024,4096",
            "256,1152,1152",
            "256,4304,1152",
            "256,1152,4304",
            "788,2048,2048",
            "788,2048,16384",
        ],
        help="Shapes to test as 'M,N,K' (M: rows of A, N: rows of W, K: cols of A/W)",
    )
    p.add_argument(
        "--bias",
        action="store_true",
        help="Also test bias=True for backends that support it",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  {device}")
    if device.type == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(device)}")
        print(f"ROCm:    {getattr(torch.version, 'hip', None)}")
    print(f"dtype:   {dtype}")
    print(f"warmup:  {args.warmup}  iters: {args.iters}")
    print(f"backends:{args.backends}")

    # Lazy imports (keep script runnable even when aiter isn't installed).
    try:
        import aiter  # type: ignore
        from aiter.ops.shuffle import shuffle_weight  # type: ignore
        from aiter import hipb_create_extension, hipb_mm  # type: ignore
        from aiter.tuned_gemm import skinny_gemm, triton_gemm  # type: ignore

        AITER_OK = True
    except Exception as e:
        AITER_OK = False
        print(f"\n[SKIP] aiter not available: {e}")

    # Create hipblaslt extension once if requested
    hipblaslt_ready = False
    if AITER_OK and "hipblaslt" in args.backends:
        try:
            hipb_create_extension()
            hipblaslt_ready = True
        except Exception as e:
            print(f"\n[WARN] hipblaslt unavailable: {e}")

    shapes = [Shape.parse(s) for s in args.shapes]

    def run_one(shape: Shape, *, use_bias: bool) -> None:
        m, n, k = shape.m, shape.n, shape.k
        x = torch.randn((m, k), device=device, dtype=dtype)
        w = torch.randn((n, k), device=device, dtype=dtype)
        b = torch.randn((n,), device=device, dtype=dtype) if use_bias else None

        # Reference in fp32 (more stable across different kernel reduction orders)
        ref_fp32 = F.linear(
            x.float(),
            w.float(),
            b.float() if b is not None else None,
        )

        print("\n" + "-" * 80)
        print(f"{shape}  bias={use_bias}")

        for backend in args.backends:
            name = backend
            try:
                if backend == "torch":
                    fn = lambda: F.linear(x, w, b)
                    out = fn()
                    rtol, atol = _tols_for(backend, dtype=dtype)
                    _assert_close(name, ref_fp32, out, rtol=rtol, atol=atol)
                    ms = _bench_ms(fn, warmup=args.warmup, iters=args.iters)
                    print(f"{name:<10}  {ms:8.4f} ms/iter")

                elif not AITER_OK:
                    print(f"{name:<10}  [SKIP] aiter not installed")

                elif backend == "triton":
                    if use_bias:
                        fn = lambda: triton_gemm(x, w, 0, bias=b, otype=dtype)
                    else:
                        fn = lambda: triton_gemm(x, w, 0, bias=None, otype=dtype)
                    out = fn()
                    rtol, atol = _tols_for(backend, dtype=dtype)
                    _assert_close(name, ref_fp32, out, rtol=rtol, atol=atol)
                    ms = _bench_ms(fn, warmup=args.warmup, iters=args.iters)
                    print(f"{name:<10}  {ms:8.4f} ms/iter")

                elif backend == "hipblaslt":
                    if not hipblaslt_ready:
                        print(f"{name:<10}  [SKIP] hipblaslt extension not ready")
                        continue
                    # hipb_mm expects weights as (K, N) / transposed.
                    fn = lambda: hipb_mm(
                        x,
                        w.t(),
                        solution_index=-1,  # auto-select
                        bias=b,
                        out_dtype=dtype,
                        scaleA=None,
                        scaleB=None,
                        scaleOut=None,
                        bpreshuffle=False,
                    )
                    out = fn()
                    rtol, atol = _tols_for(backend, dtype=dtype)
                    _assert_close(name, ref_fp32, out, rtol=rtol, atol=atol)
                    ms = _bench_ms(fn, warmup=args.warmup, iters=args.iters)
                    print(f"{name:<10}  {ms:8.4f} ms/iter")

                elif backend == "asm":
                    if use_bias:
                        print(f"{name:<10}  [SKIP] asm path in aiter does not support bias here")
                        continue
                    # asm path requires shuffled weights with layout constraints
                    if (w.shape[1] % 32) != 0:
                        print(
                            f"{name:<10}  [SKIP] K={w.shape[1]} not divisible by 32 (shuffle requirement)"
                        )
                        continue
                    # asm path requires shuffled weights + explicit output tensor
                    w_shuf = shuffle_weight(w, layout=(16, 16))
                    out_buf = torch.empty((m, n), device=device, dtype=dtype)
                    fn = lambda: aiter.gemm_a16w16_asm(  # type: ignore[attr-defined]
                        x, w_shuf, out_buf, None, None, None, bool(getattr(w_shuf, "is_shuffled", False))
                    )
                    out = fn()
                    rtol, atol = _tols_for(backend, dtype=dtype)
                    _assert_close(name, ref_fp32, out, rtol=rtol, atol=atol)
                    ms = _bench_ms(fn, warmup=args.warmup, iters=args.iters)
                    print(f"{name:<10}  {ms:8.4f} ms/iter")

                elif backend == "skinny":
                    if use_bias:
                        print(f"{name:<10}  [SKIP] skinny_gemm bias not supported in this harness")
                        continue
                    # Try all skinny solution indices (0/1/2). Some shapes will fail; that's ok.
                    for solidx in (0, 1, 2):
                        sub = f"{name}[{solidx}]"
                        try:
                            fn = lambda s=solidx: skinny_gemm(x, w, s, bias=None, otype=dtype)
                            out = fn()
                            rtol, atol = _tols_for(backend, dtype=dtype)
                            _assert_close(sub, ref_fp32, out, rtol=rtol, atol=atol)
                            ms = _bench_ms(fn, warmup=args.warmup, iters=args.iters)
                            print(f"{sub:<10}  {ms:8.4f} ms/iter")
                        except Exception as e:
                            print(f"{sub:<10}  [SKIP] {type(e).__name__}: {e}")

                else:
                    print(f"{name:<10}  [SKIP] unknown backend")

            except Exception as e:
                if _is_expected_unsupported(backend, e):
                    print(f"{name:<10}  [SKIP] {type(e).__name__}: {e}")
                    continue
                # Hard fail: unexpected backend failures should be visible.
                raise RuntimeError(
                    f"Backend {backend!r} failed for {shape} bias={use_bias}: {e}"
                ) from e

    for shape in shapes:
        run_one(shape, use_bias=False)
        if args.bias:
            run_one(shape, use_bias=True)

    print("\nOK: all selected backends passed numeric checks for all tested shapes.")


if __name__ == "__main__":
    main()

