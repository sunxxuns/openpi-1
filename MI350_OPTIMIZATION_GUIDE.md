# MI350 Optimization Guide

## Performance Target

Since MI350 draws **1000W** vs H200's **700W**, we need proportionally better performance:

| GPU | Power | Current Latency | Target Latency | Perf/Watt |
|-----|-------|-----------------|----------------|-----------|
| H200 | 700W | 32.9ms | - | 0.043 |
| MI350 (current) | 1000W | 35.7ms | - | 0.028 |
| **MI350 (target)** | **1000W** | **23ms** | ✓ | **0.043** |

**Target: 35% latency reduction (35.7ms → 23ms)**

## Current Best Results

| Configuration | Latency | Speedup | Reduction |
|--------------|---------|---------|-----------|
| Standard eager | 3.54 ms | 1.00x | baseline |
| torch.compile default | 2.75 ms | 1.29x | 22% |
| SDPA + Triton | 2.71 ms | 1.31x | 23% |
| **Aiter FA + Triton** | **2.61 ms** | **1.36x** | **26%** |
| Target | 2.30 ms | 1.54x | 35% |

**Current gap: 0.31ms (13.4% over target)**

## Policy Inference (the real workload)

The main target workload in this repo is:

- `scripts/benchmark_policy_inference.py` (Pi0 full policy inference, ~3.5B params, batch=1, 10 denoising steps)

### Current best result on MI350 (this repo)

- **~31.2 ms mean latency** using:
  - `TORCH_COMPILE_MODE=default`
  - `OPENPI_MANUAL_CUDAGRAPH=1` (manual full-call capture+replay)
  - `AITER_PRESHUFFLE_WEIGHTS=1` (pre-shuffle eligible Linear weights for bpreshuffle asm GEMM)
  - `OPENPI_DISABLE_COMPILE_AITER_ATTN=0` (compile through aiter attention; avoids graph-break kernel bloat)
  - `OPENPI_AITER_ATTN_DIRECT_MHA=1` (use `aiter.ops.mha.mha_fwd` fast path)
  - `OPENPI_INDUCTOR_MEMORY_PLANNING=0` (ROCm stability + better kernel graph)

Runbook: see `MI350_POLICY_INFERENCE_RUNBOOK.md`.

## Key Findings

### 1. torch.compile Issues on MI350

| Mode | Result | Notes |
|------|--------|-------|
| default | 1.29x | Modest improvement |
| reduce-overhead | **0.07x (65x SLOWER!)** | HIP graphs broken |
| max-autotune | 0.61x (slower) | Triton GEMM overhead |

**Recommendation: Do NOT use `reduce-overhead` or `max-autotune` on MI350**

### 1b. ROCm CUDAGraph capture with torch.compile

Manual full-call CUDAGraph replay can work on ROCm, but we found a key failure mode:

- If Dynamo traces during capture, it calls `torch.cuda.get_rng_state()` which queries CUDA generator seed.
- ROCm disallows this during capture and errors.

We apply a best-effort runtime patch in `openpi.models_pytorch.rocm_cudagraph_dynamo_patch` that
skips CUDA RNG preservation only while the stream is capturing.

### 2. Backend Performance

| Operation | rocBLAS | Triton | Winner |
|-----------|---------|--------|--------|
| GEMM | 0.10ms | 0.16ms | **rocBLAS (60% faster)** |
| RMSNorm | 0.12ms | 0.03ms | **Triton (4x faster)** |
| GELU+Mul | 0.06ms | 0.02ms | **Triton (2.5x faster)** |
| Attention | SDPA | Aiter FA | **Aiter (5% faster)** |

### 3. HIP Graphs

- Graph capture works but **adds overhead** instead of reducing it
- HIP graph replay: 0.89x (11% slower than eager!)
- Root cause: HIP runtime overhead outweighs launch savings

For the full policy benchmark, **manual full-call CUDAGraph replay** helps modestly (and is stable once capture succeeds),
but it is not sufficient to reach 23ms without additional compute-side gains.

## Recommended Configuration

```python
# Best MI350 configuration (eager mode with optimized kernels)

import torch
import torch.nn.functional as F

# 1. Import optimized kernels
from openpi.models_pytorch.triton_ops import (
    rms_norm_triton,      # 4x faster than LayerNorm
    silu_and_mul_triton,  # 2.5x faster than separate ops
)
from aiter import flash_attn_func  # Faster than SDPA

# 2. Use in your model
class OptimizedLayer(torch.nn.Module):
    def forward(self, x):
        # RMSNorm (Triton)
        h = rms_norm_triton(x, self.norm_weight)
        
        # Attention (Aiter Flash Attention)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        attn_out = flash_attn_func(q, k, v, causal=False)
        x = x + self.o_proj(attn_out)
        
        # MLP (fused Triton)
        h = rms_norm_triton(x, self.norm2_weight)
        gate_up = self.gate_up_proj(h)
        mlp_out = silu_and_mul_triton(gate_up)  # Fused!
        x = x + self.down_proj(mlp_out)
        
        return x

# 3. DO NOT use torch.compile with reduce-overhead!
# model = torch.compile(model, mode="reduce-overhead")  # BAD - 65x slower!
# Instead, just use eager mode with optimized kernels
```

## Environment Variables

```bash
# Recommended for MI350
export HIP_LAUNCH_BLOCKING=0
export AMD_LOG_LEVEL=0
export HIP_CACHE_ENABLED=1

# DO NOT enable HIP graphs (they add overhead)
export ENABLE_HIP_GRAPHS=0

# Use rocBLAS for GEMMs
export TORCH_ROCM_USE_HIPBLASLT=0
```

## Remaining Gap Analysis

Current: 2.61ms → Target: 2.30ms → **Gap: 0.31ms (13.4%)**

To close the gap, potential optimizations:
1. **Further attention optimization** - Custom Triton attention kernel
2. **Memory bandwidth** - Reduce tensor copies/allocations
3. **Kernel fusion** - Fuse more operations into single kernels
4. **Hardware tuning** - NUMA balancing, GPU clocks

## Summary

| What Works | What Doesn't Work |
|------------|-------------------|
| rocBLAS for GEMMs | Triton for GEMMs |
| Triton for elementwise | torch.compile reduce-overhead |
| Aiter Flash Attention | HIP/CUDA graphs |
| Eager mode | max-autotune mode |

**Best approach: Eager mode with Aiter FA + Triton kernels = 1.36x speedup (26% reduction)**
