# NVIDIA H200 Benchmark Results

Comparison benchmark results for NVIDIA H200 vs AMD MI350 (PR #858).

## Test Environment

- **GPU:** NVIDIA H200 (8x, 143GB HBM3e each)
- **PyTorch:** 2.10.0a0+b4e4ee81d3.nv25.12
- **CUDA:** 13.1
- **Flash Attention:** 2.7.4 (Tri Dao's implementation)
- **Attention Backends:** eager, SDPA (with Flash SDP), flash_attention_2

## Pi0 Full Policy Inference (3.5B model, batch=1)

| Metric | AMD MI350 (Aiter) | NVIDIA H200 (eager) | NVIDIA H200 (FA2) |
|--------|-------------------|---------------------|-------------------|
| Mean Latency | 142.0 ms | 120.8 ms | 118.5 ms |
| Throughput | 7.04 Hz | **8.28 Hz** | **8.44 Hz** |
| Memory | 7.10 GB | 7.06 GB | 7.06 GB |
| Denoising Steps | 10 | 10 | 10 |
| Actions Shape | (1, 10, 32) | (1, 10, 32) | (1, 10, 32) |

**Pipeline:** SigLIP image encoding → Gemma text encoding → Prefill (KV cache) → 10 denoising steps

**Result:** H200 is ~15-17% faster on inference

## 8-GPU DDP Training (3.3B Model)

| Batch/GPU | Total Batch | Seq | AMD MI350 (Aiter) | H200 (eager) | H200 (SDPA) |
|-----------|-------------|-----|-------------------|--------------|-------------|
| 4 | 32 | 512 | 225 samples/s | 212 samples/s | 218 samples/s |
| 8 | 64 | 512 | 329 samples/s | 272 samples/s | 279 samples/s |
| 8 | 64 | 1024 | 196 samples/s | 149 samples/s | 157 samples/s |
| 16 | 128 | 512 | **407 samples/s** | 312 samples/s | **320 samples/s** |

**Result:** MI350 is ~21-27% faster on training (407 vs 320 samples/s best case)

## Analysis

### Inference Performance (H200 wins by ~15-17%)
- Hopper tensor core optimizations (`nvjet_sm90_*` kernels)
- Better memory bandwidth (HBM3e vs HBM3)
- Native BF16 tensor core support

### Training Performance (MI350 wins by ~21-27%)
- AMD Aiter Flash Attention + Triton kernels optimized for training
- Better compute/memory overlap in backward pass
- Custom optimizations in PR #858 specifically target training workloads

### SDPA vs eager on H200
- SDPA gives ~2-5% improvement over eager (320 vs 312 samples/s)
- SDPA uses PyTorch's Flash SDP backend (optimized for Hopper)

## Trace Files

Available in `traces/` directory for analysis with [Perfetto](https://ui.perfetto.dev/):

| Trace | File | Size |
|-------|------|------|
| H200 Inference | `h200_inference.json` | 280.6 MB |
| H200 DDP SDPA (rank 0-7) | `h200_ddp_sdpa_rank[0-7].json` | ~41 MB each |
| H200 DDP eager (rank 0-7) | `h200_ddp_training_rank[0-7].json` | ~46 MB each |

## Running Benchmarks

### Inference Benchmark
```bash
python scripts/benchmark_policy_inference.py
```

### DDP Training Benchmark (eager)
```bash
torchrun --nproc_per_node=8 scripts/benchmark_h200_ddp.py
```

### DDP Training Benchmark (SDPA)
```bash
torchrun --nproc_per_node=8 scripts/benchmark_h200_ddp_sdpa.py
```

## Top CUDA Operations (H200 Inference)

| Operation | Self CUDA % | Self CUDA Time | Calls | GFLOPs |
|-----------|-------------|----------------|-------|--------|
| `aten::mm` | 26.65% | 54.9 ms | 6930 | 15,959 |
| `aten::bmm` | 13.20% | 27.2 ms | 2035 | 523 |
| `aten::mul` | 11.94% | 24.6 ms | 10480 | 3 |
| `aten::copy_` | 9.73% | 20.0 ms | 9105 | -- |
| `aten::add` | 9.52% | 19.6 ms | 9945 | 1 |
| `aten::addmm` | 8.76% | 18.1 ms | 2695 | 3,178 |

### Key H200-Specific Kernels (SM90/Hopper)

| Kernel | CUDA Time | Calls |
|--------|-----------|-------|
| `nvjet_sm90_tst_64x8_64x16_4x2_h_bz_TNT` | 15.9 ms | 3600 |
| `gemmSN_NN_kernel` | 14.2 ms | 900 |
| `nvjet_sm90_tst_128x272_64x4_2x1_v_bz_coopA_TNT` | 13.2 ms | 180 |
| `nvjet_sm90_tst_48x64_64x15_4x2_h_bz_bias_TNN` | 13.1 ms | 2025 |

## Conclusion

- **For inference workloads:** NVIDIA H200 provides better performance (~15-17% faster)
- **For training workloads:** AMD MI350 with Aiter+Triton optimizations provides better performance (~21-27% faster)
- The performance difference in training suggests AMD's custom kernel optimizations in PR #858 are particularly effective for backward pass operations
