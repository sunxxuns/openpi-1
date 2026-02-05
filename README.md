# Benchmark results (Pi0 policy inference, E2E)

This README intentionally contains **only** the latest benchmark results for the OpenPI **Pi0 policy inference** workload.

## Latest headline results (B=1, 10 denoising steps)

| GPU | Latency (ms) | Throughput (Hz) | Notes |
|-----|--------------|-----------------|-------|
| **NVIDIA H200** | **25.35** | **39.45** | `h200-benchmark-comparison` (`907ce98`) |
| **AMD MI350** | **21.2** | **47.15** | best-known (default benchmark; `OPENPI_SKIP_MASKED_IMAGES=1`) |
| **AMD MI350** | **24.8** | **40.29** | best-known **without** skipping masked cameras (`OPENPI_SKIP_MASKED_IMAGES=0`) |

## MI350 optimization ladder (same workload)

All numbers below are from `scripts/benchmark_policy_inference.py` on MI350 (event timing).

| Step | Enabled | Mean (ms) | Hz | Δ vs prev (ms) |
|------|---------|-----------|----|----------------|
| 0 | baseline (no manual graph, no SDPA KV-cache fast-path, no fused-linear routing, no masked-image skip) | 63.1 | 15.85 | - |
| 1 | + manual full-call CUDAGraph replay (`OPENPI_MANUAL_CUDAGRAPH=1`) | 30.1 | 33.26 | -33.0 |
| 2 | + KV-cache SDPA fast-path (`OPENPI_EAGER_ATTN_USE_SDPA=1`) | 27.3 | 36.69 | -2.8 |
| 3 | + route fused projections to aiter GEMM (`OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1`) | 26.3 | 37.97 | -1.0 |
| 4a | + skip fully-masked cameras (**no** extra tuned M=532 GEMMs) | 22.4 | 44.60 | -3.9 |
| 4b | + tuned M=532 GEMMs (`configs/openpi_bf16_tuned_gemm.csv`) | 22.1 | 45.26 | -0.3 |
| 4c | + fuse SigLIP QKV (3 GEMMs → 1) (`OPENPI_FUSE_SIGLIP_QKV=1`) | 21.4 | 46.69 | -0.7 |
| 4d | + route fused SigLIP QKV through aiter tuned GEMM (`OPENPI_ROUTE_SIGLIP_FUSED_QKV_TO_AITER=1`) | 21.2 | 47.15 | -0.2 |

## Power Efficiency Analysis

| Metric | H200 | MI350 | Gap |
|--------|------|-------|-----|
| **Throughput** | 39.45 Hz | 47.15 Hz | +20% |
| **Latency** | 25.35 ms | 21.2 ms | -16% |
| **Power** | 700 W | 1000 W | +43% |
| **Efficiency** | 0.0564 samples/J | 0.0472 samples/J | **-16%** |

### Root Cause: Kernel Fusion Gap

| Category | H200 | MI350 |
|----------|------|-------|
| Fused kernels (GEMM+GELU+Norm) | **41.2%** | 11.5% |
| Separate GEMM | 52.7% | 54.2% |
| Separate activation | 1.3% | 5.9% |
| Separate norm | 2.9% | 8.9% |

H200's `torch.compile` generates highly fused Triton kernels that combine GEMM with activation and normalization in single kernels (e.g., `triton_tem_fused__unsafe_view_gelu_mm_mul_t_view`). MI350 uses faster rocBLAS for GEMMs but cannot fuse epilogue operations, resulting in more memory traffic.

### What's Already Optimized on MI350

- ✅ HIP graph replay (100% kernel utilization, lower overhead than H200)
- ✅ Aiter Flash Attention (1.22x faster than SDPA)
- ✅ Fused GELU+mul kernel (separate from GEMM)
- ✅ Tuned rocBLAS GEMM kernels for OpenPI shapes
- ✅ Masked image skipping
- ✅ Fused SigLIP QKV projection

### Remaining Optimization Opportunities

1. **Power tuning** (Medium effort): Lower power cap to ~750W may achieve efficiency parity
2. **Custom fused Triton kernels** (High effort): GEMM+GELU fusion for MI350
3. **Batching** (User-level): Larger batch sizes improve GPU utilization
