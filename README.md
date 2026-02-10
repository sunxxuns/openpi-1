# OpenPI Policy Inference Benchmark — AMD MI300X (BSZ=1)


## Models

| Model | Params | Architecture |
|-------|-------:|--------------|
| **Pi0** | 3.5B | SigLIP ViT + Gemma 2B + Gemma 300M action expert |
| **Pi0.5** | 3.5B | SigLIP ViT + Gemma 2B + Gemma 300M action expert (AdaRMS) |

Both models: 3 camera images (224×224), 10 diffusion denoising steps.


## End-to-end latency (BSZ=1, CUDAGraph, fastest)

### Pi0

| GPU | Arch | E2E latency (ms) | Throughput (Hz) |
|-----|------|------------------:|----------------:|
| **AMD MI300X** | gfx942 | **26.1** | **38.3** |
| AMD MI350 | gfx950 | 25.3 | 39.5 |
| NVIDIA H200 | sm_90 | 25.35 | 39.5 |

### Pi0.5

| GPU | Arch | E2E latency (ms) | Throughput (Hz) |
|-----|------|------------------:|----------------:|
| **AMD MI300X** | gfx942 | **28.7** | **34.9** |

Pi0.5 is ~10% slower than Pi0 due to AdaRMS conditioning in the action expert.


## Latency breakdown (BSZ=1, MI300X, from profiler trace)

### Pi0

Total GPU kernel time: **27.1 ms**. CUDAGraph replay: **26.1 ms**.

| Stage | GPU time (ms) | % of total | Key ops |
|-------|-------------:|-----------:|---------|
| **ViT** (SigLIP) | **4.1** | **15%** | GEMM 3.0 + FlashAttn 0.9 + GELU 0.2 |
| **LLM prefill** (Gemma 2B) | **10.7** | **39%** | GEMM 8.3 + aiter MHA 1.7 + GELU 0.5 + RMSNorm 0.3 |
| **Diffusion** (10 denoise steps) | **11.2** | **41%** | GEMM 5.1 + SDPA 1.5 + RoPE 2.9 + GELU 0.4 + RMSNorm 0.8 + other 0.6 |
| Other | 1.1 | 4% | fill, misc |
| **Total** | **27.1** | **100%** | |

Per denoise step: **~1.1 ms**.

### Pi0.5

Total GPU kernel time: **30.9 ms**. CUDAGraph replay: **28.7 ms**.

| Stage | GPU time (ms) | % of total | Key ops |
|-------|-------------:|-----------:|---------|
| **ViT** (SigLIP) | **4.1** | **13%** | Same as Pi0 |
| **LLM prefill** (Gemma 2B) | **10.7** | **34%** | Same as Pi0 |
| **Diffusion + AdaRMS** (10 denoise steps) | **13.6** | **44%** | GEMM 5.8 + SDPA 1.6 + RoPE 2.9 + **AdaRMS 1.0** + other 2.3 |
| Other | 2.6 | 8% | fill, misc |
| **Total** | **30.9** | **100%** | |

Per denoise step: **~1.4 ms** (vs Pi0's ~1.1 ms). The +0.3 ms/step is from AdaRMS conditioning.

**Pi0 vs Pi0.5 delta** (27.1 → 30.9 ms = **+3.8 ms**):
- AdaRMS conditioning: **+1.0 ms** (new `triton_per_fused_addmm_silu`, 27 calls)
- Increased fill/misc overhead: **+2.8 ms** (more buffers for AdaRMS state)


### Pi0 breakdown details

**ViT (SigLIP)** — 3 cameras × 9 layers, 27 flash-attention calls:
- `aten::addmm` (biased linear, SigLIP-only): 2.978 ms / 130 calls
- `aten::_flash_attention_forward` [3,256,16,72]: 0.896 ms / 27 calls
- `triton_poi_fused_gelu_9`: 0.190 ms / 27 calls

**LLM prefill (Gemma 2B)** — 18 layers, M=788 prefix tokens, aiter flash attention:
- `aten::mm` (fused QKV/Gate+Up/Down, M=788): 8.260 ms / 73 calls
- `aiter::mha_fwd` [1,788,8,256]: 1.652 ms / 18 calls
- `triton_poi_fused_gelu_mul_2` [1,788,32768]: 0.493 ms / 18 calls
- `triton_red_fused_rmsnorm_0` [1,788,2048]: 0.268 ms / 37 calls

**Diffusion (10 denoise steps)** — each step runs Gemma 2B decode (M=11 suffix, KV-cached) + Gemma 300M expert:
- `aten::mm` (small M=11 GEMMs): 5.058 ms / 720 calls
- `aten::_efficient_attention_forward` [1,11,8,256]: 1.488 ms / 180 calls
- Triton fused RoPE/reshape (180 calls each): 2.934 ms
- `triton_poi_fused_gelu_mul_16`: 0.399 ms / 180 calls
- RMSNorm (expert + decode): 0.765 ms / 312 calls
- `aten::fill_` + other: 0.587 ms


## MI350 results (from `wip/mi350-23ms-fusion-tuning` branch)

### Pi0 BSZ=1

| GPU | Arch | E2E latency (ms) | Throughput (Hz) |
|-----|------|------------------:|----------------:|
| AMD MI350 | gfx950 | 25.3 | 39.5 |
| NVIDIA H200 | sm_90 | 25.35 | 39.5 |

MI350 trace: 1522 kernels, 15.0 ms kernel sum, 100% GPU utilization.

### MI350 batch size scaling (Pi0)

| BSZ | MI350 Latency (ms) | MI350 Samples/s | H200 Latency (ms) | H200 Samples/s | MI350 speedup |
|-----|--------------------:|----------------:|-------------------:|---------------:|--------------:|
| 1 | 25.3 | 39.5 | 25.3 | 39.5 | 1.00x |
| 2 | 36.9 | 54.2 | 34.4 | 58.1 | 0.93x |
| 4 | 49.4 | 81.0 | 53.4 | 74.9 | 1.08x |
| 8 | 71.5 | 111.9 | 94.1 | 85.0 | 1.32x |
| 16 | 121.9 | 131.2 | 175.1 | 91.4 | 1.44x |
| 32 | 217.8 | 146.9 | 340.5 | 94.0 | 1.56x |
| **64** | **416.3** | **153.7** | **655.9** | **97.6** | **1.58x** |

### MI350 optimization ladder (Pi0)

| Step | Enabled | Mean (ms) | Hz | Δ vs prev |
|------|---------|-----------|----|-----------|
| 0 | baseline | 63.1 | 15.9 | - |
| 1 | + CUDAGraph replay | 30.1 | 33.3 | -33.0 |
| 2 | + SDPA KV-cache fast-path | 27.3 | 36.7 | -2.8 |
| 3 | + route fused projections to aiter GEMM | 26.3 | 38.0 | -1.0 |
| 4 | + fuse SigLIP QKV (3 GEMMs → 1) | 25.8 | 38.8 | -0.5 |
| 5 | + native GELU + aggressive fusion | 25.3 | 39.5 | -0.5 |


## Traces

Viewable at [ui.perfetto.dev](https://ui.perfetto.dev/).

| File | Model | HW | Method | Kernels | GPU time | Size |
|------|-------|----|--------|--------:|--------:|-----:|
| `mi300x_pi0_bsz1_cudagraph_rocprof_26ms.json` | Pi0 | MI300X | rocprofv3 (CUDAGraph) | 2929 | 29.9 ms | 517 KB |
| `mi300x_pi0_bsz1_no_cudagraph_27ms.json` | Pi0 | MI300X | PyTorch profiler | 2927 | 27.1 ms | 18 MB |
| `mi300x_pi05_bsz1_cudagraph_rocprof_28ms.json` | Pi0.5 | MI300X | rocprofv3 (CUDAGraph) | 3061 | 32.1 ms | 535 KB |
| `mi300x_pi05_bsz1_no_cudagraph_31ms.json` | Pi0.5 | MI300X | PyTorch profiler | 3061 | 30.9 ms | 19 MB |
| `mi350_policy_inference_21ms_47hz.json` | Pi0 | MI350 | PyTorch profiler (CUDAGraph) | 1522 | 15.0 ms | 844 KB |
| `h200_policy_inference_25ms_39hz.json` | Pi0 | H200 | PyTorch profiler | — | — | 3.4 MB |

The MI300X CUDAGraph trace was captured using `rocprofv3 --kernel-trace` which instruments at the HIP level and sees every GPU kernel inside graph replay. The ~3 ms delta (29.9 vs 27.1 ms) is rocprofv3 per-kernel instrumentation overhead.


## How to reproduce

```bash
cd /sgl-workspace/openpi-1

# Pi0 BSZ=1 (CUDAGraph, fastest)
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

# Pi0.5 BSZ=1 (CUDAGraph, fastest)
OPENPI_PI05=1 \
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

# CUDAGraph trace via rocprofv3
AITER_PRESHUFFLE_WEIGHTS=0 OPENPI_SKIP_MASKED_IMAGES=0 \
OPENPI_EAGER_ATTN_USE_SDPA=1 OPENPI_FUSE_SIGLIP_QKV=1 \
OPENPI_ROUTE_SIGLIP_FUSED_QKV_TO_AITER=1 \
OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1 OPENPI_ROUTE_FUSED_LINEAR_M_THRESH=1000000 \
TORCH_COMPILE_MODE=default \
rocprofv3 --kernel-trace -d traces/rocprof -o mi300x_pi0_graph -- \
python scripts/trace_cudagraph.py

# Extract last graph replay to Chrome trace
python scripts/extract_rocprof_trace.py traces/rocprof/mi300x_pi0_graph_results.db \
  -o traces/mi300x_pi0_bsz1_cudagraph_rocprof_26ms.json
```

## Environment

- AMD MI300X (gfx942), 304 CUs, 206 GB HBM3
- PyTorch 2.9.0a0+git7bcbafe, ROCm/HIP 7.0
- aiter flash attention + aiter GEMM (hipblaslt)
- torch.compile mode=default, manual CUDAGraph capture
- `OPENPI_SKIP_MASKED_IMAGES=0` (apples-to-apples with H200)
