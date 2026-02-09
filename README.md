# Pi0 / Pi0.5 Policy Inference — AMD MI350 vs NVIDIA H200

## Models

| Model | Architecture | Params | Description |
|-------|-------------|--------|-------------|
| **Pi0** | SigLIP ViT + Gemma-2B + Gemma-300M expert | 3.5B | Flow-matching VLA |
| **Pi0.5** | SigLIP ViT + Gemma-2B + Gemma-300M expert (adaRMS) | 3.6B | Upgraded Pi0 with adaptive RMSNorm |

**Workload**: 3 camera images (224×224) → SigLIP ViT → Gemma-2B prefill (788 tokens) → 10-step denoising (Gemma-2B decode + expert) → 32-dim actions. BF16. Identical on both GPUs.


## B=1 Inference Results

| Model | GPU | Power (W) | Latency (ms) | Throughput (Hz) | vs H200 |
|-------|-----|-----------|--------------|-----------------|---------|
| **Pi0** | **MI350** | **700** | **23.7** | **42.2** | $\color{green}{\textbf{1.07x}}$ |
| Pi0 | H200 | 700 | 25.35 | 39.45 | baseline |
| **Pi0.5** | **MI350** | **700** | **24.8** | **39.9** | — |

MI350 @ 700W beats H200 @ 700W: **23.7 ms vs 25.35 ms** at the same power budget.


## Latency Breakdown (B=1, CUDAGraph replay, rocprof kernel trace, MI350 @ 700W)

| Category | Pi0 (ms) | Pi0 (%) | Pi0.5 (ms) | Pi0.5 (%) |
|----------|---------|---------|-----------|-----------|
| **GEMM** (rocBLAS/hipBLASLt) | 14.0 | 58.0% | 13.7 | 53.8% |
| **Fused ops** (Triton: rotary, GELU, SDPA pre/post) | 4.5 | 18.7% | 6.1 | 23.8% |
| **Attention** (aiter MHA + SDPA + CK fmha) | 3.0 | 12.5% | 3.0 | 11.9% |
| **RMSNorm** (Triton reduction) | 1.8 | 7.6% | 1.9 | 7.6% |
| **Fill/elementwise** | 0.7 | 2.7% | 0.7 | 2.6% |
| **Total GPU compute** | **24.1** | | **25.6** | |
| **Kernels in graph** | 2909 | | 3034 | |
| **E2E (CUDA event)** | **23.7** | | **24.8** | |

Pi0.5 is 1.5ms slower: GEMMs are similar but **Triton fused ops grow +1.6ms** from adaRMS conditioning (extra dense projections + element-wise modulation per expert layer → more Inductor-generated Triton kernels).

GEMMs are **58%** of Pi0 compute, **54%** of Pi0.5. Attention is **12%**. Triton fused ops are **19-24%**.

Traces viewable in `chrome://tracing`:
- `traces/mi350_cudagraph_replay_rocprof.json` (Pi0, 596 KB)
- `traces/mi350_cudagraph_replay_pi05_rocprof.json` (Pi0.5, 492 KB)

Note: traces show ~38-41ms wall due to rocprof per-kernel instrumentation overhead (~5us × 3000 kernels). The actual CUDAGraph replay has zero inter-kernel gaps. Kernel durations are accurate.


## Pi0 Batch Size Scaling (throughput, MI350 @ 700W)

| BSZ | MI350 Samples/s | H200 Samples/s | MI350 vs H200 |
|-----|----------------:|---------------:|---------------:|
| 1 | 42.2 | 39.5 | $\color{green}{\textbf{1.07x}}$ |
| 2 | 64.4 | 58.1 | $\color{green}{\textbf{1.11x}}$ |
| 4 | 96.6 | 74.9 | $\color{green}{\textbf{1.29x}}$ |
| 8 | 132.4 | 85.0 | $\color{green}{\textbf{1.56x}}$ |
| 16 | 156.6 | 91.4 | $\color{green}{\textbf{1.71x}}$ |
| 32 | 168.6 | 94.0 | $\color{green}{\textbf{1.79x}}$ |
| **64** | **182.4** | **97.6** | $\color{green}{\textbf{1.87x}}$ |

## Pi0 Batch Size Scaling (latency, MI350 @ 700W)

| BSZ | MI350 (ms) | H200 (ms) |
|-----|----------:|---------:|
| 1 | 23.7 | 25.3 |
| 2 | 31.1 | 34.4 |
| 4 | 41.4 | 53.4 |
| 8 | 59.9 | 94.1 |
| 16 | 100.9 | 175.1 |
| 32 | 189.8 | 340.5 |
| 64 | 349.8 | 655.9 |

MI350 peak: **182.4 samples/s** at BSZ 64 ($\color{green}{\textbf{1.87x}}$ H200)


## Reproduce

All optimizations are defaulted in the code:

```bash
cd /sgl-workspace/openpi

# Pi0, B=1
python scripts/benchmark_policy_inference.py

# Pi0.5, B=1
OPENPI_PI05=1 python scripts/benchmark_policy_inference.py

# Pi0, batch sweep
bash scripts/run_bsz_sweep.sh
```

Key optimizations baked into defaults:
- **Batched SigLIP**: 3 camera images in one forward pass
- **CUDAGraph**: captures full `sample_actions()` for zero CPU overhead
- **Inductor kernel tuning**: coordinate descent, benchmark kernel, group fusion, weight freezing
- **Fused projections**: QKV and Gate+Up as single GEMMs via aiter
- **SDPA KV-cache attention**: efficient attention kernels for decode
- **aiter flash attention**: direct `mha_fwd` for prefill, compiled through Inductor


## Optimization Ladder (Pi0)

| Step | Optimization | Latency (ms) | Hz | Delta |
|------|-------------|--------------|-----|-------|
| 0 | baseline | 63.1 | 15.9 | - |
| 1 | + CUDAGraph replay | 30.1 | 33.3 | -33.0 |
| 2 | + SDPA KV-cache attention | 27.3 | 36.7 | -2.8 |
| 3 | + fused projections via aiter GEMM | 26.3 | 38.0 | -1.0 |
| 4 | + SigLIP QKV fusion | 25.8 | 38.8 | -0.5 |
| 5 | + native GELU + aggressive Inductor fusion | 25.3 | 39.5 | -0.5 |
| 6 | + batched SigLIP + Inductor kernel tuning | 23.7 | 42.2 | -1.6 |
