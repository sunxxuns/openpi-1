# Benchmark results (Pi0 policy inference, E2E)


## Latest headline results (B=1, 10 denoising steps)

| GPU | Power (W) | Latency (ms) | Throughput (Hz) |
|-----|-----------|--------------|-----------------|
| **NVIDIA H200** | 700 | **25.35** | **39.45** |
| **AMD MI350** | 1000 | **23.7** | **42.2** |


## Batch size scaling

All optimizations enabled (batched SigLIP, Inductor kernel tuning, CUDAGraph). 10 denoising steps. H200 @ 700W, MI350 @ 1000W.

| BSZ | MI350 Latency (ms) | MI350 Samples/s | H200 Latency (ms) | H200 Samples/s | MI350 speedup |
|-----|--------------------:|----------------:|-------------------:|---------------:|--------------:|
| 1 | 23.7 | 42.2 | 25.3 | 39.5 | 1.07x |
| 2 | 31.1 | 64.4 | 34.4 | 58.1 | 1.11x |
| 4 | 41.4 | 96.6 | 53.4 | 74.9 | 1.29x |
| 8 | 59.9 | 132.4 | 94.1 | 85.0 | 1.56x |
| 16 | 100.9 | 156.6 | 175.1 | 91.4 | 1.71x |
| 32 | 189.8 | 168.6 | 340.5 | 94.0 | 1.79x |
| **64** | **349.8** | **182.4** | **655.9** | **97.6** | **1.87x** |


MI350 peak: **182.4 samples/s** at BSZ 64 (1.87x H200 peak of 97.6)

## MI350 optimization ladder

All numbers below are from `scripts/benchmark_policy_inference.py` on MI350 (wall timing, B=1).

| Step | Enabled | Mean (ms) | Hz | Δ vs prev (ms) |
|------|---------|-----------|----|----------------|
| 0 | baseline (no manual graph, no SDPA KV-cache fast-path, no fused-linear routing) | 63.1 | 15.85 | - |
| 1 | + manual full-call CUDAGraph replay (`OPENPI_MANUAL_CUDAGRAPH=1`) | 30.1 | 33.26 | -33.0 |
| 2 | + KV-cache SDPA fast-path (`OPENPI_EAGER_ATTN_USE_SDPA=1`) | 27.3 | 36.69 | -2.8 |
| 3 | + route fused projections to aiter GEMM (`OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1`) | 26.3 | 37.97 | -1.0 |
| 4 | + fuse SigLIP QKV (3 GEMMs → 1) (`OPENPI_FUSE_SIGLIP_QKV=1`) | 25.8 | 38.76 | -0.5 |
| 5 | + native GELU + aggressive Inductor fusion (`OPENPI_NATIVE_GELU=1`, `OPENPI_AGGRESSIVE_FUSION=1`) | 25.3 | 39.47 | -0.5 |
| 6 | + batched SigLIP + Inductor kernel tuning (coordinate descent, benchmark kernel, group fusion, freezing) | 23.7 | 42.2 | -1.6 |
