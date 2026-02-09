# Benchmark results (Pi0 policy inference, E2E)


## Latest headline results (B=1, 10 denoising steps)

| GPU | Power (W) | Latency (ms) | Throughput (Hz) |
|-----|-----------|--------------|-----------------|
| **NVIDIA H200** | 700 | **25.35** | **39.45** |
| **AMD MI350** | 1000 | **25.3** | **39.47** |
| **AMD MI300X** | 750 | **26.6** | **37.64** |


## Batch size scaling

All optimizations enabled. 10 denoising steps. No masked-image skipping (apples-to-apples).

| BSZ | MI300X Latency (ms) | MI300X Samples/s | MI350 Latency (ms) | MI350 Samples/s | H200 Latency (ms) | H200 Samples/s |
|-----|--------------------:|----------------:|--------------------:|----------------:|-------------------:|---------------:|
| 1 | 26.6 | 37.6 | 25.3 | 39.5 | 25.3 | 39.5 |
| 2 | 36.3 | 55.1 | 36.9 | 54.2 | 34.4 | 58.1 |
| 4 | 54.3 | 73.7 | 49.4 | 81.0 | 53.4 | 74.9 |
| 8 | 91.2 | 87.7 | 71.5 | 111.9 | 94.1 | 85.0 |
| 16 | 165.1 | 96.9 | 121.9 | 131.2 | 175.1 | 91.4 |
| 32 | 307.9 | 104.0 | 217.8 | 146.9 | 340.5 | 94.0 |
| **64** | **581.4** | **110.1** | **416.3** | **153.7** | **655.9** | **97.6** |

MI300X peak: **110.1 samples/s** at BSZ 64 (1.13x H200 peak of 97.6)
MI350 peak: **153.7 samples/s** at BSZ 64 (1.58x H200 peak of 97.6)


## MI300X vs MI350 vs H200 notes

- **MI300X (gfx942)**: 304 CUs, 206 GB HBM3, ~5.3 TB/s BW, ~750W TDP
- **MI350 (gfx950)**: 304 CUs, 192 GB HBM3, ~8 TB/s BW, ~1000W TDP
- **H200**: 132 SMs, 141 GB HBM3e, ~4.8 TB/s BW, ~700W TDP

At BSZ 1 (real-time robot control):
- MI300X is ~5% slower than MI350/H200 in latency, but draws ~25% less power
- MI300X perf/watt: 37.64 Hz / 750W = **0.050 samples/s/W**
- MI350 perf/watt: 39.47 Hz / 1000W = 0.039 samples/s/W
- H200 perf/watt: 39.45 Hz / 700W = 0.056 samples/s/W

At BSZ 64 (peak throughput):
- MI300X trails MI350 by ~28% in throughput (lower memory bandwidth)
- MI300X still beats H200 by 13% in peak throughput

Key differences MI300 vs MI350:
- No gfx950 ASM GEMM kernels (uses hipblaslt fallback)
- No weight pre-shuffle (`AITER_PRESHUFFLE_WEIGHTS=0`)
- Lower memory bandwidth limits batch scaling advantage


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
