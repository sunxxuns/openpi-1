# Task: Add AMD MI300/MI350 GPU Backend Support for Pi0 Policy Inference

## Overview

This repository contains a Pi0/Pi0.5 policy inference model for robot control. The model currently runs on NVIDIA GPUs. Your task is to add AMD MI300X and MI350 GPU backend support so that the full inference pipeline runs correctly and efficiently on AMD hardware.

## What the Model Does

Pi0 is a vision-language-action (VLA) model for robot control:
- **Input**: 3 camera images (224x224) + language prompt + robot state
- **Pipeline**: SigLIP ViT image encoding -> Gemma-2B prefill (788 tokens) -> 10-step denoising (Gemma-2B + Gemma-300M expert) -> 32-dim actions
- **Output**: Action trajectory (batch_size x action_horizon x action_dim), all in BF16
- **Target**: Real-time robot control at batch_size=1

## Hardware Targets

- **AMD Instinct MI300X** (gfx942, 192GB HBM3, 304 CUs)
- **AMD Instinct MI350** (gfx950, 288GB HBM3e, 320 CUs)

Both GPUs use ROCm and HIP (AMD's CUDA-compatible runtime).

## Benchmark Scripts

Two benchmark scripts are provided. **These scripts should NOT be modified** — they test the model's public API (`model.sample_actions()`). All optimizations should happen inside the model code.

### 1. `scripts/benchmark_policy_inference.py` — Performance Benchmark

Measures end-to-end inference latency at batch_size=1.

```bash
# Basic benchmark
python scripts/benchmark_policy_inference.py

# With CUDA event timing (measures GPU time directly)
python scripts/benchmark_policy_inference.py --timing cuda_event

# With CUDAGraph capture+replay (reduces CPU launch overhead)
python scripts/benchmark_policy_inference.py --cudagraph

# With profiling (exports chrome trace)
python scripts/benchmark_policy_inference.py --profile
```

**Expected results after successful AMD support:**
- The script runs to completion without errors
- Latency is reported (reasonable range: 20-100ms depending on optimization level)
- Actions tensor has shape `(1, 10, 32)`

### 2. `scripts/verify_precision.py` — Correctness Verification

Verifies that inference produces valid and deterministic outputs.

```bash
python scripts/verify_precision.py
```

**Expected results after successful AMD support:**
- `[PASS] No NaN values in outputs`
- `[PASS] No Inf values in outputs`
- `[PASS] Output shape: (1, 10, 32)`
- `[PASS] Deterministic` (or `[WARN] Near-deterministic` for bf16)
- `[PASS] Output values in reasonable range`
- `RESULT: ALL CHECKS PASSED`

## Design Principles for AMD Support

1. **Do not modify the benchmark scripts.** They test the public API `model.sample_actions()`. Optimizations should be implemented at lower abstraction layers.

2. **The model uses `torch.compile`** (applied in `PI0Pytorch.__init__`). On AMD/ROCm, you may need to adjust the compile mode or handle ROCm-specific compilation issues inside the model code.

3. **Key optimization areas** (for achieving competitive latency):
   - Attention kernels (e.g., flash attention implementations for ROCm)
   - GEMM tuning (finding optimal matrix multiply kernels for the model's hot shapes)
   - CUDAGraph support (reducing CPU launch overhead on ROCm)
   - Fused operations (reducing kernel launch count)
   - Inductor backend tuning for ROCm

4. **AMD-specific libraries** that may be useful:
   - `aiter`: AMD's inference/training engine with optimized attention and GEMM kernels
   - `Composable Kernel (CK)`: AMD's kernel library
   - `hipBLASLt` / `rocBLAS`: AMD's BLAS libraries
   - Triton: Works on ROCm for custom fused kernels

## Success Criteria

1. **Correctness**: `verify_precision.py` reports ALL CHECKS PASSED
2. **Performance**: `benchmark_policy_inference.py` completes and reports reasonable latency
   - Baseline (unoptimized): ~60-100ms is acceptable as a first pass
   - Optimized target: ~25ms (competitive with NVIDIA H200)
3. **No benchmark modification**: The benchmark scripts run unmodified
