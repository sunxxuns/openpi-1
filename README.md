# OpenPI Policy Inference Benchmark — AMD MI300X

## Model

**Pi0** (`pi05=False`), 3.5B parameters.  
Architecture: PaliGemma (SigLIP ViT + Gemma 2B) + Gemma 300M action expert.

## BSZ=1 headline (10 denoising steps)

| GPU | Arch | Latency (ms) | Hz | Power (W) | Perf/W (Hz/kW) |
|-----|------|-------------:|---:|----------:|-----------:|
| **AMD MI300X** | gfx942 | **26.4** | **37.9** | ~750 | **50.5** |
| AMD MI350 | gfx950 | 25.3 | 39.5 | ~1000 | 39.5 |
| NVIDIA H200 | sm_90 | 25.3 | 39.5 | ~700 | 56.4 |

MI300X is ~4% slower than MI350/H200 in raw latency but ~28% better perf/watt than MI350.

## BSZ=1 latency breakdown (CUDAGraph, fastest)

CUDAGraph replay total: **26.4 ms** (37.9 Hz).

Proportional breakdown estimated from per-stage CUDA-event timing (steady-state, iterations 14–30):

| Stage | Without graph (ms) | % of total | Estimated with graph (ms) |
|-------|-------------------:|-----------:|--------------------------:|
| **ViT** (SigLIP vision, 3 cameras) | 8.1 | 5.9% | ~1.6 |
| **LLM prefill** (Gemma 2B, KV-cache build) | 15.3 | 11.2% | ~2.9 |
| **Diffusion** (10 denoise steps, Gemma 300M expert) | 113.5 | 82.9% | ~21.9 |
| **Total** | **136.9** | **100%** | **26.4** |

Per denoise step: ~2.2 ms (with graph) / ~11.4 ms (without graph).

The 5× speedup from CUDAGraph comes from eliminating kernel launch overhead (~110 ms of CPU-side dispatch is replaced by a single graph replay).

## How to reproduce

```bash
cd /sgl-workspace/openpi-1

# BSZ=1 with CUDAGraph (headline number)
AITER_PRESHUFFLE_WEIGHTS=0 \
OPENPI_SKIP_MASKED_IMAGES=0 \
OPENPI_MANUAL_CUDAGRAPH=1 \
OPENPI_EAGER_ATTN_USE_SDPA=1 \
OPENPI_FUSE_SIGLIP_QKV=1 \
OPENPI_ROUTE_SIGLIP_FUSED_QKV_TO_AITER=1 \
OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1 \
OPENPI_ROUTE_FUSED_LINEAR_M_THRESH=1000000 \
TORCH_COMPILE_MODE=default \
python scripts/benchmark_policy_inference.py --batch-size 1

# Latency breakdown (ViT / LLM / Diffusion)
AITER_PRESHUFFLE_WEIGHTS=0 \
OPENPI_SKIP_MASKED_IMAGES=0 \
OPENPI_EAGER_ATTN_USE_SDPA=1 \
OPENPI_FUSE_SIGLIP_QKV=1 \
OPENPI_ROUTE_SIGLIP_FUSED_QKV_TO_AITER=1 \
OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1 \
OPENPI_ROUTE_FUSED_LINEAR_M_THRESH=1000000 \
TORCH_COMPILE_MODE=default \
python scripts/benchmark_policy_inference_breakdown.py
```

## MI300X vs MI350 key differences

| | MI300X (gfx942) | MI350 (gfx950) |
|---|---|---|
| CUs | 304 | 304 |
| Memory | 206 GB HBM3 | 192 GB HBM3 |
| Bandwidth | ~5.3 TB/s | ~8 TB/s |
| TDP | ~750W | ~1000W |
| ASM GEMM kernels | No (hipblaslt only) | Yes (aiter gfx950 ASM) |
| Weight pre-shuffle | No (`AITER_PRESHUFFLE_WEIGHTS=0`) | Available |

## Batch size scaling (reference)

All optimizations enabled, 10 denoising steps, no masked-image skip.

| BSZ | MI300X (ms) | MI300X Samples/s | MI350 (ms) | MI350 Samples/s | H200 (ms) | H200 Samples/s |
|-----|------------:|----------------:|-----------:|----------------:|----------:|---------------:|
| 1 | 26.6 | 37.6 | 25.3 | 39.5 | 25.3 | 39.5 |
| 2 | 36.3 | 55.1 | 36.9 | 54.2 | 34.4 | 58.1 |
| 4 | 54.3 | 73.7 | 49.4 | 81.0 | 53.4 | 74.9 |
| 8 | 91.2 | 87.7 | 71.5 | 111.9 | 94.1 | 85.0 |
| 16 | 165.1 | 96.9 | 121.9 | 131.2 | 175.1 | 91.4 |
| 32 | 307.9 | 104.0 | 217.8 | 146.9 | 340.5 | 94.0 |
| **64** | **581.4** | **110.1** | **416.3** | **153.7** | **655.9** | **97.6** |

MI300X peak: 110.1 samples/s at BSZ 64 (1.13× H200).

## Environment

- PyTorch 2.9.0a0+git7bcbafe, ROCm/HIP 7.0
- aiter flash attention + aiter GEMM (hipblaslt, no ASM)
- torch.compile mode=default, manual CUDAGraph capture
- `OPENPI_SKIP_MASKED_IMAGES=0` (apples-to-apples with H200)
