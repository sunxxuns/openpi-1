# MI350 Performance Analysis - Deep Dive

## Executive Summary

**Problem**: MI350 runs at ~1000W vs H200 ~700W on this workload; we want perf/W parity.
**Target**: 23ms latency to match H200 perf/W (\(700/1000 \times 32.9\text{ms} \approx 23.0\text{ms}\)).
**Achieved (policy inference E2E)**: **~21.2ms mean** on MI350 for the default end-to-end policy inference benchmark by
skipping fully-masked cameras (drop their image tokens) + **SigLIP QKV fusion** + manual full-call CUDAGraph replay + SDPA KV-cache fast-path + aiter tuned GEMM.
**Note**: If all 3 images are present (no fully-masked camera), best-known is closer to **~24.8–25.0ms** and remains compute-bound.

### Update: policy inference benchmark (real workload)

For the actual OpenPI Pi0 end-to-end policy inference benchmark (`scripts/benchmark_policy_inference.py`),
we improved the MI350 result from **35.7ms** (torch.compile baseline) to **~21.2ms** by combining:

- Manual full-call **CUDAGraph capture+replay** (`OPENPI_MANUAL_CUDAGRAPH=1`)
- **aiter tuned GEMM** routing for `nn.Linear`
- **KV-cache attention fast-path** using SDPA (`OPENPI_EAGER_ATTN_USE_SDPA=1`)
- **Skip fully-masked cameras** and drop their image tokens (`OPENPI_SKIP_MASKED_IMAGES=1`)
 - **SigLIP QKV fusion** (`OPENPI_FUSE_SIGLIP_QKV=1`)

This **does** reach the 23ms perf/watt target for the default benchmark (where at least one camera is fully masked).
If all cameras are present, the best-known config is still ~26ms and is primarily **GEMM/compute bound**.

## Root Cause (historical): HIP graph overhead on compiled subgraphs

### The Core Issue

HIP/CUDA graphs can eliminate launch overhead, but the benefit depends heavily on **what** you capture
(full-call replay vs capturing already-fused compiled subgraphs) and whether capture introduces extra work.

| Test | Eager | HIP Graph | Speedup |
|------|-------|-----------|---------|
| 100 elementwise ops | 0.56ms | 0.19ms | **2.93x** ✓ |
| 50 GEMM ops | 0.61ms | 0.31ms | **1.96x** ✓ |
| 4-layer transformer | 3.56ms | 3.56ms | **1.00x** ✗ |
| torch.compile | 2.76ms | 3.78ms | **0.73x** ✗ |

**Key Finding**: HIP graphs help for many small ops. Capturing/replaying **already-fused compiled subgraphs**
can add overhead and/or force fallbacks. In contrast, capturing a **full-call** inference step and replaying it
(`OPENPI_MANUAL_CUDAGRAPH=1`) is still a large win for end-to-end policy inference.

### Why HIP Graphs Don't Help Transformers

1. **hipBLASLt Graph Capture Issue** (FIXED in ROCm 7.0)
   - hipBLASLt now supports stream capture with `DISABLE_ADDMM_CUDA_LT=1`
   - But this uses slower GEMM fallback

2. **HIP Graph Replay Overhead**
   - HIP graph replay has ~10-15μs fixed overhead per replay
   - For small ops: overhead < launch savings → speedup
   - For large fused ops: overhead > launch savings → slowdown

3. **torch.compile Already Fuses**
   - Reduces kernel count by 3-5x through Triton fusion
   - Fewer kernels → less launch overhead to save
   - Graph capture adds overhead without proportional benefit

## Performance Breakdown

### Current Best Configuration

| Configuration | Latency | Speedup |
|--------------|---------|---------|
| Eager (baseline) | 3.54ms | 1.00x |
| + Custom Triton kernels | 3.38ms | 1.05x |
| + SDPA attention | 2.71ms | 1.31x |
| + Aiter Flash Attention | 2.61ms | **1.36x** |
| torch.compile | 2.76ms | 1.29x |
| Target (H200 parity) | 2.30ms | 1.54x |

### Where Time is Spent (Compiled Model)

| Operation | % of CUDA Time |
|-----------|---------------|
| GEMM (rocBLAS) | 79% |
| Flash Attention | 8% |
| LayerNorm/Triton | 3% |
| Other | 10% |

**GEMM is 79% of compute and already at 90% efficiency** - no room to improve.

## What We Tried

### ✅ Works Well
- **Triton fused kernels**: RMSNorm 4x faster, SiLU+Mul 2.5x faster
- **Aiter Flash Attention**: 5% faster than SDPA
- **torch.compile default mode**: 1.29x speedup through fusion

### ✗ Doesn't Help
- **HIP graphs on compiled code**: 0.73x (27% slower!)
- **torch.compile reduce-overhead**: 65x slower (broken)
- **Aggressive fusion settings**: No improvement over default
- **max-autotune mode**: Same performance, longer compile

### ⚠️ Fundamental Limitations
- **hipBLASLt graph overhead**: Fixed in ROCm but still slower
- **HIP graph replay overhead**: ~15μs per replay, can't be reduced
- **GEMM efficiency**: Already at 90%, near theoretical peak

## Recommendations

### For Best Performance Today (policy inference E2E)

Use the best-known end-to-end config from `MI350_BEST_KNOWN_CONFIG.md` / `MI350_POLICY_INFERENCE_RUNBOOK.md`.
In short: `torch.compile` (default) + manual full-call CUDAGraph replay + SDPA KV-cache fast-path + aiter tuned GEMM,
and `OPENPI_SKIP_MASKED_IMAGES=1` when cameras are fully masked.

### Do NOT Use
```python
# BAD - 65x slower on MI350!
model = torch.compile(model, mode="reduce-overhead")

# Capturing the wrong granularity can add overhead; prefer full-call capture+replay.
```

### For AMD to Fix
1. **Reduce HIP graph replay overhead** from ~15μs to <5μs
2. **hipBLASLt graph support** without falling back to slower GEMM
3. **Better Triton GEMM kernels** to match rocBLAS performance

## Conclusion

For the default OpenPI policy inference benchmark, MI350 **can** match H200's perf/watt target at BF16
by removing overhead and (critically) skipping compute for fully-masked cameras (dropping their image tokens).

| Metric | H200 | MI350 | Gap |
|--------|------|-------|-----|
| Power | 700W | 1000W | +43% |
| Latency | 32.9ms | 35.7ms | +8% |
| Best optimized (default benchmark) | - | **~22.2ms** | **-38%** |
| Best optimized (all cameras present) | - | ~26ms | -27% |

When all cameras are present, the remaining gap vs the 23ms target is primarily compute/GEMM-bound and may
require further kernel-level improvements (or future ROCm stack improvements).

### Path Forward
1. **Accept current perf/watt gap** - MI350 at ~88% of H200 efficiency
2. **Wait for ROCm improvements** - HIP graph overhead reduction
3. **Consider FP8 quantization** - 1.57x GEMM speedup available with Aiter
