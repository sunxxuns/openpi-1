# Benchmark results (Pi0 policy inference, E2E)


## Latest headline results (B=1, 10 denoising steps)

| GPU | Power (W) | Latency (ms) | Throughput (Hz) |
|-----|-----------|--------------|-----------------|
| **NVIDIA H200** | 700 | **25.35** | **39.45** |
| **AMD MI350** | 1000 | **25.3** | **39.47** |


## Batch size scaling

All optimizations enabled. 10 denoising steps. H200 @ 700W, MI350 @ 1000W.

| BSZ | MI350 Latency (ms) | MI350 Samples/s | H200 Latency (ms) | H200 Samples/s | MI350 speedup |
|-----|--------------------:|----------------:|-------------------:|---------------:|--------------:|
| 1 | 25.3 | 39.5 | 25.3 | 39.5 | 1.00x |
| 2 | 36.9 | 54.2 | 34.4 | 58.1 | 0.93x |
| 4 | 49.4 | 81.0 | 53.4 | 74.9 | 1.08x |
| 8 | 71.5 | 111.9 | 94.1 | 85.0 | 1.32x |
| 16 | 121.9 | 131.2 | 175.1 | 91.4 | 1.44x |
| 32 | 217.8 | 146.9 | 340.5 | 94.0 | 1.56x |
| **64** | **416.3** | **153.7** | **655.9** | **97.6** | **1.58x** |


MI350 peak: **153.7 samples/s** at BSZ 64 (1.58x H200 peak of 97.6)

## MI350 optimization ladder

All numbers below are from `scripts/benchmark_policy_inference.py` on MI350 (event timing, B=1).

| Step | Enabled | Mean (ms) | Hz | Δ vs prev (ms) |
|------|---------|-----------|----|----------------|
| 0 | baseline (no manual graph, no SDPA KV-cache fast-path, no fused-linear routing) | 63.1 | 15.85 | - |
| 1 | + manual full-call CUDAGraph replay (`OPENPI_MANUAL_CUDAGRAPH=1`) | 30.1 | 33.26 | -33.0 |
| 2 | + KV-cache SDPA fast-path (`OPENPI_EAGER_ATTN_USE_SDPA=1`) | 27.3 | 36.69 | -2.8 |
| 3 | + route fused projections to aiter GEMM (`OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1`) | 26.3 | 37.97 | -1.0 |
| 4 | + fuse SigLIP QKV (3 GEMMs → 1) (`OPENPI_FUSE_SIGLIP_QKV=1`) | 25.8 | 38.76 | -0.5 |
| 5 | + native GELU + aggressive Inductor fusion (`OPENPI_NATIVE_GELU=1`, `OPENPI_AGGRESSIVE_FUSION=1`) | 25.3 | 39.47 | -0.5 |
