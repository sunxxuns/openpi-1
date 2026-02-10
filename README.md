# OpenPI Policy Inference Benchmark — AMD MI300X (BSZ=1)

## Model

**Pi0** (`pi05=False`), 3.5B parameters.  
- Vision: SigLIP ViT (3 camera images, 224×224)  
- Language: Gemma 2B (prefix / KV-cache build)  
- Action expert: Gemma 300M (10 diffusion denoising steps)

## End-to-end latency (BSZ=1, CUDAGraph, fastest)

| GPU | Arch | E2E latency (ms) | Throughput (Hz) |
|-----|------|------------------:|----------------:|
| **AMD MI300X** | gfx942 | **26.4** | **37.9** |
| AMD MI350 | gfx950 | 25.3 | 39.5 |
| NVIDIA H200 | sm_90 | 25.3 | 39.5 |

## Latency breakdown (BSZ=1, CUDAGraph)

Total with CUDAGraph replay: **26.4 ms**.

| Stage | Time (ms) | % of E2E | Notes |
|-------|----------:|---------:|-------|
| **ViT** (SigLIP) | **1.6** | 6% | 3 cameras → image tokens |
| **LLM prefill** (Gemma 2B) | **2.9** | 11% | KV-cache build from image+text prefix |
| **Diffusion** (Gemma 300M expert) | **21.9** | 83% | 10 denoise steps × ~2.2 ms/step |
| **E2E total** | **26.4** | 100% | |

Diffusion dominates at 83% of E2E. Per denoise step: **~2.2 ms**.

Methodology: per-stage GPU time measured with CUDA events under `torch.compile` (steady-state iterations 14–30), then proportionally mapped to the CUDAGraph E2E total. Individual stages cannot be timed inside a graph replay since the entire `sample_actions` call is captured as a single graph.

## How to reproduce

```bash
cd /sgl-workspace/openpi-1

# E2E latency (CUDAGraph, fastest)
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

# Per-stage breakdown (ViT / LLM / Diffusion)
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

## Environment

- AMD MI300X (gfx942), 304 CUs, 206 GB HBM3
- PyTorch 2.9.0a0+git7bcbafe, ROCm/HIP 7.0
- aiter flash attention + aiter GEMM (hipblaslt)
- torch.compile mode=default, manual CUDAGraph capture
- `OPENPI_SKIP_MASKED_IMAGES=0` (apples-to-apples with H200)
