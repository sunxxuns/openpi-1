## MI350 OpenPI policy inference runbook (portable)

This is the **one command** you want on a fresh machine to reproduce the current best end-to-end latency for the Pi0 policy inference benchmark.

### Goal

- **Current best (this repo, MI350)**: ~**23.7 ms** mean for `scripts/benchmark_policy_inference.py` (B1, 10 denoising steps, all 3 cameras active)
- **H200 reference**: **25.3 ms** (same workload)
- MI350 beats H200 by **1.07x** at B=1, scaling to **1.87x** at B=64

### What actually worked

- **Keep BF16** (no FP8/INT8 requirements for this path)
- **Batch SigLIP** (key win for all-cameras-active workload):
  - `OPENPI_BATCH_SIGLIP=1` batches all camera images into one SigLIP forward pass (M=768 instead of 3 × M=256). Same FLOPs, better GPU utilization.
- **Fuse SigLIP QKV projections** (consistent win):
  - `OPENPI_FUSE_SIGLIP_QKV=1` fuses SigLIP vision tower Q/K/V projections (3 GEMMs → 1 GEMM).
- **torch.compile**: use `TORCH_COMPILE_MODE=default` on ROCm
- **Manual full-call CUDAGraph replay**: capture+replay `PI0Pytorch.sample_actions(...)`
  - ROCm capture can fail if Dynamo traces during capture because Dynamo saves CUDA RNG state.
  - We patch Dynamo at runtime (best-effort) to skip CUDA RNG get_state while a stream capture is active.
- **Inductor kernel tuning** (collectively ~1.6ms win):
  - `OPENPI_COORD_DESCENT=1`: coordinate descent tuning for Triton kernel configs
  - `OPENPI_BENCHMARK_KERNEL=1`: benchmark kernel implementations per op
  - `OPENPI_GROUP_FUSION=1`: fuse groups of similar operations
  - `OPENPI_MAX_FUSION_SIZE=128`: allow larger fused kernels
  - `OPENPI_FREEZING=1`: constant-fold weights for inference
- **GEMM**:
  - route `nn.Linear` to **aiter tuned GEMM** dispatcher
  - route **fused QKV + fused Gate+Up** projections through aiter tuned GEMM (so they use our tuned configs too)
  - route the **fused SigLIP QKV** projection through aiter tuned GEMM:
    - `OPENPI_ROUTE_SIGLIP_FUSED_QKV_TO_AITER=1`
  - keep **global bpreshuffle off** (`AITER_PRESHUFFLE_WEIGHTS=0`) for best end-to-end latency
  - Note: ASM bshuffle kernels are 20-32% faster than rocBLAS for M=788 shapes in isolation,
    but aiter's custom-op dispatch overhead under torch.compile + CUDAGraph currently negates the gain.
    Tuned configs are in `configs/openpi_bf16_tuned_gemm.csv` for future use when dispatch is improved.
- **Attention (important)**: compile *through* aiter attention with the direct `mha_fwd` path.
  - `OPENPI_DISABLE_COMPILE_AITER_ATTN=0`
  - `OPENPI_AITER_ATTN_DIRECT_MHA=1`
  - `OPENPI_INDUCTOR_MEMORY_PLANNING=0` (Inductor stability on ROCm; also reduces kernel count)
- **KV-cache attention fast-path** (big win): enable SDPA for the `q_len != k_len` case to avoid explicit
  `bmm + softmax + bmm` and use the efficient attention kernels:
  - `OPENPI_EAGER_ATTN_USE_SDPA=1`

### Recommended environment

```bash
export HIP_LAUNCH_BLOCKING=0
export AMD_LOG_LEVEL=0
export HIP_CACHE_ENABLED=1
```

### Best known command (end-to-end)

```bash
cd /sgl-workspace/openpi

AITER_PRESHUFFLE_WEIGHTS=0 \
OPENPI_SKIP_MASKED_IMAGES=0 \
OPENPI_MANUAL_CUDAGRAPH=1 \
OPENPI_EAGER_ATTN_USE_SDPA=1 \
OPENPI_FUSE_SIGLIP_QKV=1 \
OPENPI_ROUTE_SIGLIP_FUSED_QKV_TO_AITER=1 \
OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1 \
OPENPI_ROUTE_FUSED_LINEAR_M_THRESH=1000000 \
TORCH_COMPILE_MODE=default \
OPENPI_BATCH_SIGLIP=1 \
OPENPI_COORD_DESCENT=1 \
OPENPI_BENCHMARK_KERNEL=1 \
OPENPI_GROUP_FUSION=1 \
OPENPI_MAX_FUSION_SIZE=128 \
OPENPI_FREEZING=1 \
python scripts/benchmark_policy_inference.py
```

Expected: **~23.7 ms** mean, **~42.2 Hz** throughput (on MI350), with all 3 cameras active.

### Notes / caveats

- **Memory**: `AITER_PRESHUFFLE_WEIGHTS=0` keeps memory lower and avoids regressions for small-M GEMMs.
- **First run**: aiter may JIT/build some modules on first execution; Inductor coordinate descent tuning also adds to first compile time. Benchmark uses warmup, but overall script runtime may be longer the first time.
- **If capture fails**: the script prints a warning and falls back to normal execution. If that happens, rerun with a larger warmup, or disable manual graph and use compiled-only mode.
- **If you see ~28ms**: you likely forgot the Inductor tuning knobs (OPENPI_COORD_DESCENT, OPENPI_BENCHMARK_KERNEL, etc.) or OPENPI_BATCH_SIGLIP.
- **If you see ~34-35ms**: you likely re-enabled one of these (unset them):
  - `OPENPI_DISABLE_COMPILE_AITER_ATTN=1` (graph-breaks around attention; increases kernel count)
  - `OPENPI_INDUCTOR_MEMORY_PLANNING=1` (can trigger Inductor issues on ROCm / regresses kernel graph)
