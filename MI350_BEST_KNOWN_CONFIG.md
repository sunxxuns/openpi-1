## MI350 Best Known Config (do not regress)

This file exists to prevent "we forgot the magic env vars" regressions.

### Current best result (policy inference E2E)

- **~23.7 ms mean** (≈ **42.2 Hz**) with all 3 cameras active (no skip-masked-images):
  - `OPENPI_BATCH_SIGLIP=1` (batch 3 images into one SigLIP forward pass)
  - `OPENPI_EAGER_ATTN_USE_SDPA=1`
  - `OPENPI_FUSE_SIGLIP_QKV=1` (SigLIP vision tower: fuse Q/K/V projections; 3 GEMMs → 1 GEMM)
  - `OPENPI_COORD_DESCENT=1` (Inductor coordinate descent kernel tuning)
  - `OPENPI_BENCHMARK_KERNEL=1` (Inductor benchmarks kernel implementations per op)
  - `OPENPI_GROUP_FUSION=1` (Inductor fuses groups of similar operations)
  - `OPENPI_MAX_FUSION_SIZE=128` (larger fused kernels, fewer HIP graph launches)
  - `OPENPI_FREEZING=1` (constant-fold weights for inference)
  - tuned bf16 GEMM configs in `configs/openpi_bf16_tuned_gemm.csv`
  - fused projections routed through aiter tuned GEMM
- **~182.4 samples/s** peak throughput at BSZ 64 (1.87x H200)

### What to run (policy inference E2E)

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

### The critical invariants (do not regress)

These must remain true (either by script defaults or by explicit env vars):

- `OPENPI_DISABLE_COMPILE_AITER_ATTN=0`
  - If set to `1`, we graph-break around aiter attention and regress to ~34ms due to more kernels.
- `OPENPI_AITER_ATTN_DIRECT_MHA=1`
  - Forces the direct `torch.ops.aiter.mha_fwd` call (more `torch.compile` friendly).
- `OPENPI_INDUCTOR_MEMORY_PLANNING=0`
  - Avoids Inductor ROCm recursion issues and keeps the compiled graph/kernel-count low.
- `OPENPI_EAGER_ATTN_USE_SDPA=1`
  - Enables an SDPA fast-path for the **q_len != k_len** KV-cache attention case that otherwise falls back to
    explicit `bmm + softmax + bmm`. This is the largest single win.
- `OPENPI_BATCH_SIGLIP=1`
  - Batch all active camera images into a single SigLIP forward pass (M=768 instead of 3 × M=256).
    Same FLOPs, better GPU utilization, fewer kernel launches.
- `OPENPI_FUSE_SIGLIP_QKV=1`
  - Fuses SigLIP vision tower Q/K/V projections (3 GEMMs → 1 GEMM).
- `AITER_PRESHUFFLE_WEIGHTS=0`
  - Global bpreshuffle duplicates weights (memory) and can regress small-M decode-ish GEMMs; best-known config keeps it off.
- `OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1` (and a large `..._M_THRESH`)
  - Ensures fused QKV + fused Gate+Up projections use aiter tuned GEMM (and our tuned configs).
- `OPENPI_ROUTE_SIGLIP_FUSED_QKV_TO_AITER=1`
  - Routes the **fused SigLIP QKV** projection through aiter GEMM so it can use the tuned config entry for
    the \(M=256, N=3456, K=1152\) shape.
- `OPENPI_COORD_DESCENT=1`
  - Inductor coordinate descent tuning for Triton kernel configs (block sizes, num_warps). Longer first
    compile but better steady-state kernels.
- `OPENPI_BENCHMARK_KERNEL=1`
  - Inductor benchmarks different kernel implementations per op and picks the fastest.
- `OPENPI_GROUP_FUSION=1`
  - Inductor fuses groups of similar operations together.
- `OPENPI_FREEZING=1`
  - Enables constant-folding / weight freezing for inference.

### How to profile / confirm

```bash
PROFILE=1 PROFILE_GRAPH_REPLAY=0 PROFILE_DIR=traces/comp_best_repro \
AITER_PRESHUFFLE_WEIGHTS=0 OPENPI_MANUAL_CUDAGRAPH=1 OPENPI_EAGER_ATTN_USE_SDPA=1 \
OPENPI_FUSE_SIGLIP_QKV=1 OPENPI_ROUTE_SIGLIP_FUSED_QKV_TO_AITER=1 \
OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1 OPENPI_ROUTE_FUSED_LINEAR_M_THRESH=1000000 \
TORCH_COMPILE_MODE=default OPENPI_BATCH_SIGLIP=1 OPENPI_COORD_DESCENT=1 \
OPENPI_BENCHMARK_KERNEL=1 OPENPI_GROUP_FUSION=1 OPENPI_MAX_FUSION_SIZE=128 \
OPENPI_FREEZING=1 \
python scripts/benchmark_policy_inference.py
```
