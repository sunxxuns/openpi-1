# OpenPI Policy Inference Benchmark — AMD MI300X (BSZ=1)

## Model

**Pi0** (`pi05=False`), 3.5B parameters.  
- Vision: SigLIP ViT (3 camera images, 224×224)  
- Language: Gemma 2B (prefix / KV-cache build)  
- Action expert: Gemma 300M (10 diffusion denoising steps)

## End-to-end latency (BSZ=1, CUDAGraph, fastest)

| GPU | Arch | E2E latency (ms) | Throughput (Hz) |
|-----|------|------------------:|----------------:|
| **AMD MI300X** | gfx942 | **26.1** | **38.3** |
| AMD MI350 | gfx950 | 25.3 | 39.5 |
| NVIDIA H200 | sm_90 | 25.3 | 39.5 |

## Latency breakdown (BSZ=1, from profiler trace with CUDAGraph)

Total GPU kernel time: **27.1 ms**. CUDAGraph replay: **26.1 ms**.

| Stage | GPU time (ms) | % of total | Key ops |
|-------|-------------:|-----------:|---------|
| **ViT** (SigLIP) | **4.1** | **15%** | GEMM 3.0 + FlashAttn 0.9 + GELU 0.2 |
| **LLM prefill** (Gemma 2B) | **10.7** | **39%** | GEMM 8.3 + aiter MHA 1.7 + GELU 0.5 + RMSNorm 0.3 |
| **Diffusion** (10 denoise steps) | **11.2** | **41%** | GEMM 5.1 + SDPA 1.5 + RoPE 2.9 + GELU 0.4 + RMSNorm 0.8 + other 0.6 |
| Other | 1.1 | 4% | fill, misc |
| **Total** | **27.1** | **100%** | |

Per denoise step: **~1.1 ms**.

Trace files (viewable at [ui.perfetto.dev](https://ui.perfetto.dev/)):
- `traces/mi300x_pi0_bsz1_no_cudagraph_27ms.json` — **full kernel detail** (18 MB, 2927 kernels, 27.1 ms GPU time). Use this for breakdown.
- `traces/mi300x_pi0_bsz1_cudagraph_replay_partial_1ms.json` — graph replay, partial capture (66 KB, 79/~1500 kernels). ROCm 7.0 limitation.

ROCm 7.0 limitation: `hipGraphLaunch` kernel tracing is incomplete (only ~5% of kernels visible via `enable_cuda_sync_events`; rocprofv3 captures 0%). The MI350 trace (ROCm 6.x) does not have this issue. The non-graph trace captures the **identical kernels** — only CPU dispatch overhead differs (CUDAGraph eliminates it: ~141 ms wall → 26.1 ms).

### Breakdown details

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

# Generate trace + breakdown
PROFILE=1 PROFILE_DIR=traces \
AITER_PRESHUFFLE_WEIGHTS=0 OPENPI_SKIP_MASKED_IMAGES=0 \
OPENPI_MANUAL_CUDAGRAPH=1 OPENPI_EAGER_ATTN_USE_SDPA=1 \
OPENPI_FUSE_SIGLIP_QKV=1 OPENPI_ROUTE_SIGLIP_FUSED_QKV_TO_AITER=1 \
OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1 OPENPI_ROUTE_FUSED_LINEAR_M_THRESH=1000000 \
TORCH_COMPILE_MODE=default \
python scripts/benchmark_policy_inference.py --batch-size 1

# Analyze trace
python scripts/analyze_policy_trace.py traces/mi300x_pi0_policy_inference_26ms_38hz.json
```

## Environment

- AMD MI300X (gfx942), 304 CUs, 206 GB HBM3
- PyTorch 2.9.0a0+git7bcbafe, ROCm/HIP 7.0
- aiter flash attention + aiter GEMM (hipblaslt)
- torch.compile mode=default, manual CUDAGraph capture
- `OPENPI_SKIP_MASKED_IMAGES=0` (apples-to-apples with H200)
