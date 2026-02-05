# Benchmark results (Pi0 policy inference, E2E)

This README intentionally contains **only** the latest benchmark results for the OpenPI **Pi0 policy inference** workload.

## Latest headline results (B=1, 10 denoising steps)

| GPU | Power (W) | Latency (ms) | Throughput (Hz) | Notes |
|-----|-----------|--------------|-----------------|-------|
| **NVIDIA H200** | 700 | **25.35** | **39.45** | `h200-benchmark-comparison` |
| **AMD MI350** | 1000 | **21.2** | **47.15** | best-known (`OPENPI_SKIP_MASKED_IMAGES=1`) |
| **AMD MI350** | 1000 | **24.8** | **40.29** | best-known **without** skipping masked cameras |

## Batch size scaling

All optimizations enabled. 10 denoising steps. H200 @ 700W, MI350 @ 1000W.

| BSZ | MI350 Latency (ms) | MI350 Samples/s | H200 Latency (ms) | H200 Samples/s | MI350 speedup |
|-----|--------------------:|----------------:|-------------------:|---------------:|--------------:|
| 1 | 21.7 | 46.2 | 25.3 | 39.5 | 1.17x |
| 2 | 29.6 | 67.7 | 34.4 | 58.1 | 1.16x |
| 4 | 43.5 | 92.0 | 53.4 | 74.9 | 1.23x |
| 8 | 55.2 | 145.0 | 94.1 | 85.0 | 1.71x |
| 16 | 86.9 | 184.1 | 175.1 | 91.4 | 2.01x |
| 32 | 160.6 | 199.2 | 340.5 | 94.0 | 2.12x |
| **64** | **298.7** | **214.3** | **655.9** | **97.6** | **2.20x** |
| 128 | OOM (kernel fault) | - | 1638.4 | 78.1 | - |

MI350 peak: **214.3 samples/s** at BSZ 64 (2.20x H200 peak of 97.6)

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
