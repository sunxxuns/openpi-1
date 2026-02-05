# Benchmark results (Pi0 policy inference, E2E)

This README intentionally contains **only** the latest benchmark results for the OpenPI **Pi0 policy inference** workload.

## Latest headline results (B=1, 10 denoising steps)

| GPU | Latency (ms) | Throughput (Hz) | Notes |
|-----|--------------|-----------------|-------|
| **NVIDIA H200** | **25.35** | **39.45** | `h200-benchmark-comparison` |
| **AMD MI350** | **21.2** | **47.15** | best-known (`OPENPI_SKIP_MASKED_IMAGES=1`) |
| **AMD MI350** | **24.8** | **40.29** | best-known **without** skipping masked cameras |

## Batch size scaling

All optimizations enabled. 10 denoising steps. MI350 @ 700W power cap.

| BSZ | MI350 Latency (ms) | MI350 Samples/s | H200 Latency (ms) | H200 Samples/s |
|-----|--------------------:|----------------:|-------------------:|---------------:|
| 1 | 21.5 | 46.6 | 25.3 | 39.5 |
| 2 | 31.0 | 64.6 | 34.4 | 58.1 |
| 4 | 44.8 | 89.4 | 53.4 | 74.9 |
| 8 | 66.5 | 120.4 | 94.1 | 85.0 |
| 16 | 114.5 | 139.7 | 175.1 | 91.4 |
| 32 | 214.9 | 148.9 | 340.5 | 94.0 |
| **64** | **404.3** | **158.3** | **655.9** | **97.6** |
| 128 | OOM (kernel fault) | - | 1638.4 | 78.1 |

MI350 peak: **158.3 samples/s** at BSZ 64 (1.62x H200 peak of 97.6)

## MI350 optimization ladder

All numbers below are from `scripts/benchmark_policy_inference.py` on MI350 (event timing).

| Step | Enabled | Mean (ms) | Hz | Δ vs prev (ms) |
|------|---------|-----------|----|----------------|
| 0 | baseline (no manual graph, no SDPA KV-cache fast-path, no fused-linear routing, no masked-image skip) | 63.1 | 15.85 | - |
| 1 | + manual full-call CUDAGraph replay (`OPENPI_MANUAL_CUDAGRAPH=1`) | 30.1 | 33.26 | -33.0 |
| 2 | + KV-cache SDPA fast-path (`OPENPI_EAGER_ATTN_USE_SDPA=1`) | 27.3 | 36.69 | -2.8 |
| 3 | + route fused projections to aiter GEMM (`OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1`) | 26.3 | 37.97 | -1.0 |
| 4a | + skip fully-masked cameras (**no** extra tuned M=532 GEMMs) | 22.4 | 44.60 | -3.9 |
| 4b | + tuned M=532 GEMMs (`configs/openpi_bf16_tuned_gemm.csv`) | 22.1 | 45.26 | -0.3 |
| 4c | + fuse SigLIP QKV (3 GEMMs -> 1) (`OPENPI_FUSE_SIGLIP_QKV=1`) | 21.4 | 46.69 | -0.7 |
| 4d | + route fused SigLIP QKV through aiter tuned GEMM (`OPENPI_ROUTE_SIGLIP_FUSED_QKV_TO_AITER=1`) | 21.2 | 47.15 | -0.2 |

## Kernel Fusion Gap

| Category | H200 | MI350 |
|----------|------|-------|
| Fused kernels (GEMM+GELU+Norm) | **41.2%** | 11.5% |
| Separate GEMM | 52.7% | 54.2% |
| Separate activation | 1.3% | 5.9% |
| Separate norm | 2.9% | 8.9% |

H200's `torch.compile` generates fused Triton kernels (GEMM+activation+norm in one kernel). MI350 uses faster rocBLAS for GEMMs but cannot fuse epilogue operations.

## What's Already Optimized on MI350

- HIP graph replay (100% kernel utilization, lower overhead than H200)
- Aiter Flash Attention (1.22x faster than SDPA)
- Fused GELU+mul kernel (separate from GEMM)
- Tuned rocBLAS GEMM kernels for OpenPI shapes
- Masked image skipping
- Fused SigLIP QKV projection
