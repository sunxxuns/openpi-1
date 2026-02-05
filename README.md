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

