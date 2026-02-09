#!/usr/bin/env python3
"""Full GEMM tuner: hipBLASLt + rocBLAS + Triton + ASM for OpenPI hot shapes.

Safely enumerates all hipBLASLt/rocBLAS solver indices and benchmarks aiter's
Triton GEMM kernel to find the absolute fastest config per shape.
"""
import csv, os, sys, traceback
import torch

os.environ["AMD_LOG_LEVEL"] = "0"
os.environ["HIP_LAUNCH_BLOCKING"] = "0"

GPU = int(os.environ.get("OPENPI_GPU_ID", "7"))
DEV = torch.device(f"cuda:{GPU}")
torch.cuda.set_device(DEV)

from aiter import hipb_create_extension, hipb_mm
from aiter.ops.gradlib import hipb_findallsols, rocb_mm, rocb_findallsols
from aiter.jit.utils.chip_info import get_cu_num

try:
    from aiter import gemm_a16w16_asm
    HAS_ASM = True
except Exception:
    HAS_ASM = False

try:
    from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

hipb_create_extension()
CU = get_cu_num()
WARMUP, ITERS = 40, 200

# Hot shapes: (M, N, K, bias) — focus on low-efficiency prefill GEMMs
SHAPES = [
    (788, 2048, 2048, False),   # output proj — 36% eff
    (788, 2560, 2048, False),   # QKV fused — 37% eff
    (788, 2048, 16384, False),  # down proj — 50% eff
    (788, 32768, 2048, False),  # gate+up — 92% eff (already good, but check)
]


def _sync():
    torch.cuda.current_stream().synchronize()


def bench(fn, iters=ITERS):
    """Returns us/call."""
    for _ in range(WARMUP):
        fn()
    _sync()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def bench_rocblas(M, N, K):
    """torch.mm → rocBLAS."""
    a = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=DEV)
    return bench(lambda: torch.mm(a, b)), "torch", 0, 0


def bench_hipblaslt(M, N, K):
    """Enumerate all hipBLASLt solutions, benchmark top candidates."""
    a = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    # hipb_findallsols expects (M,K) × (K,N) → (M,N)
    b_find = torch.randn(K, N, dtype=torch.bfloat16, device=DEV)
    # hipb_mm expects (M,K) × (N,K) and does mat1 @ mat2.T → (M,N)
    b_mm = b_find.t().contiguous()
    try:
        sols = hipb_findallsols(a, b_find, out_dtype=torch.bfloat16)
    except Exception as exc:
        print(f"    hipBLASLt findallsols failed: {exc}")
        return None, None, None, None

    print(f"    hipBLASLt: {len(sols)} solutions")
    best_us, best_sol = float("inf"), -1
    tested = 0
    for sol in sols:
        try:
            out = hipb_mm(a, b_mm, sol, out_dtype=torch.bfloat16)
            if out.shape != (M, N):
                continue
            us = bench(lambda s=sol: hipb_mm(a, b_mm, s, out_dtype=torch.bfloat16), iters=80)
            tested += 1
            if us < best_us:
                best_us, best_sol = us, sol
        except Exception:
            continue
    if best_sol < 0:
        return None, None, None, None
    # Re-bench winner with more iters
    best_us = bench(lambda: hipb_mm(a, b_mm, best_sol, out_dtype=torch.bfloat16))
    print(f"    hipBLASLt best: sol={best_sol} -> {best_us:.1f} us (tested {tested})")
    return best_us, "hipblaslt", best_sol, 0


def bench_rocblas_sols(M, N, K):
    """Enumerate rocBLAS solutions via rocb_findallsols."""
    a = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=DEV)  # rocb_mm uses (K,N)
    try:
        sols = rocb_findallsols(a, b)
    except Exception as exc:
        print(f"    rocBLAS findallsols failed: {exc}")
        return None, None, None, None

    print(f"    rocBLAS: {len(sols)} solutions")
    best_us, best_sol = float("inf"), -1
    for sol in sols:
        try:
            out = rocb_mm(a, b, sol)
            if out.shape != (M, N):
                continue
            us = bench(lambda s=sol: rocb_mm(a, b, s), iters=80)
            if us < best_us:
                best_us, best_sol = us, sol
        except Exception:
            continue
    if best_sol < 0:
        return None, None, None, None
    best_us = bench(lambda: rocb_mm(a, b, best_sol))
    print(f"    rocBLAS best: sol={best_sol} -> {best_us:.1f} us")
    return best_us, "rocblas", best_sol, 0


def bench_triton_gemm(M, N, K):
    """aiter Triton GEMM kernel."""
    if not HAS_TRITON:
        return None, None, None, None
    a = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    # Triton GEMM expects weight in (N, K) layout
    w = torch.randn(N, K, dtype=torch.bfloat16, device=DEV)
    bias = torch.zeros(N, dtype=torch.bfloat16, device=DEV)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
    try:
        gemm_a16w16(a, w, bias, torch.bfloat16, out)
        us = bench(lambda: gemm_a16w16(a, w, bias, torch.bfloat16, out))
        print(f"    Triton GEMM: {us:.1f} us")
        return us, "triton", 0, 0
    except Exception as exc:
        print(f"    Triton GEMM failed: {exc}")
        return None, None, None, None


def bench_asm(M, N, K):
    """ASM GEMM kernels."""
    if not HAS_ASM or N % 64 != 0 or K % 64 != 0:
        return None, None, None, None
    a = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    b = torch.randn(N, K, dtype=torch.bfloat16, device=DEV)
    best_us, best_sol, best_sk = float("inf"), -1, 1
    for sk in [1, 2, 4]:
        for sol in range(16):
            try:
                out = gemm_a16w16_asm(a, b, sol, sk)
                if out is None or out.shape != (M, N):
                    continue
                us = bench(lambda s=sol, k=sk: gemm_a16w16_asm(a, b, s, k), iters=80)
                if us < best_us:
                    best_us, best_sol, best_sk = us, sol, sk
            except Exception:
                continue
    if best_sol < 0:
        return None, None, None, None
    best_us = bench(lambda: gemm_a16w16_asm(a, b, best_sol, best_sk))
    print(f"    ASM best: sol={best_sol} splitK={best_sk} -> {best_us:.1f} us")
    return best_us, "asm", best_sol, best_sk


def tune(M, N, K, bias):
    print(f"\n{'='*65}")
    flops = 2.0 * M * N * K
    print(f"  M={M} N={N} K={K}  ({flops/1e9:.1f} GFLOPS)")
    print(f"{'='*65}")

    results = []

    # 1. rocBLAS baseline (torch.mm)
    us, lib, sol, sk = bench_rocblas(M, N, K)
    tflops = flops / (us * 1e-6) / 1e12
    print(f"    torch.mm (rocBLAS): {us:.1f} us ({tflops:.0f} TFLOPS)")
    results.append((us, lib, sol, sk, ""))

    # 2. hipBLASLt full sweep
    us2, lib2, sol2, sk2 = bench_hipblaslt(M, N, K)
    if us2 is not None:
        results.append((us2, lib2, sol2, sk2, ""))

    # 3. rocBLAS solution sweep
    us3, lib3, sol3, sk3 = bench_rocblas_sols(M, N, K)
    if us3 is not None:
        results.append((us3, lib3, sol3, sk3, ""))

    # 4. Triton GEMM
    us4, lib4, sol4, sk4 = bench_triton_gemm(M, N, K)
    if us4 is not None:
        results.append((us4, lib4, sol4, sk4, ""))

    # 5. ASM kernels
    us5, lib5, sol5, sk5 = bench_asm(M, N, K)
    if us5 is not None:
        results.append((us5, lib5, sol5, sk5, ""))

    # Pick winner
    winner = min(results, key=lambda x: x[0])
    w_us, w_lib, w_sol, w_sk, w_kn = winner
    w_tflops = flops / (w_us * 1e-6) / 1e12
    speedup = results[0][0] / w_us
    print(f"\n  >>> WINNER: {w_lib} sol={w_sol} splitK={w_sk}")
    print(f"      {w_us:.1f} us | {w_tflops:.0f} TFLOPS | {speedup:.2f}x vs rocBLAS")

    return {
        "cu_num": CU, "M": M, "N": N, "K": K,
        "bias": bias, "dtype": "torch.bfloat16", "outdtype": "torch.bfloat16",
        "scaleAB": False, "bpreshuffle": False,
        "libtype": w_lib, "solidx": w_sol, "splitK": w_sk,
        "us": round(w_us, 4), "kernelName": w_kn,
        "err_ratio": 0.0, "tflops": round(w_tflops, 2),
        "bw": round((M*K + N*K + M*N) * 2 / (w_us * 1e-6) / 1e9, 2),
    }


def main():
    print(f"Full GEMM Tuner — hipBLASLt + rocBLAS + Triton + ASM")
    print(f"Device: cuda:{GPU} | CUs: {CU}")
    print(f"ASM: {HAS_ASM} | Triton: {HAS_TRITON}")

    results = []
    for M, N, K, bias in SHAPES:
        try:
            r = tune(M, N, K, bias)
            if r["libtype"] != "torch":
                results.append(r)
        except Exception:
            traceback.print_exc()
            continue

    # Write to CSV
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "openpi_bf16_tuned_gemm.csv"))
    existing = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = list(csv.DictReader(f))

    new_keys = {(str(r["M"]), str(r["N"]), str(r["K"])) for r in results}
    merged = [r for r in existing if (r.get("M"), r.get("N"), r.get("K")) not in new_keys]
    fnames = ["cu_num","M","N","K","bias","dtype","outdtype","scaleAB","bpreshuffle",
              "libtype","solidx","splitK","us","kernelName","err_ratio","tflops","bw"]
    for r in results:
        merged.append({k: r[k] for k in fnames})

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fnames)
        w.writeheader()
        for row in merged:
            w.writerow(row)

    print(f"\n{'='*65}")
    print(f"Wrote {len(results)} new + {len(merged)-len(results)} existing = {len(merged)} total")
    print(f"Output: {out_path}")

    tmp = "/tmp/aiter_configs/bf16_tuned_gemm.csv"
    if os.path.exists(tmp):
        os.remove(tmp)
        print(f"Cleared cache: {tmp}")


if __name__ == "__main__":
    main()
