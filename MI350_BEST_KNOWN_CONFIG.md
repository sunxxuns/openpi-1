## MI350 Best Known Config (do not regress)

This file exists to prevent “we forgot the magic env vars” regressions.

### Current best result (policy inference E2E)

- **~22.2 ms mean** (≈ **45.0 Hz**) with:
  - `OPENPI_SKIP_MASKED_IMAGES=1` (skip fully-masked cameras; drops their image tokens)
  - `OPENPI_EAGER_ATTN_USE_SDPA=1`
  - tuned bf16 GEMM configs (including fused-projection shapes and the M=532 variants in `configs/openpi_bf16_tuned_gemm.csv`)
  - fused projections routed through aiter tuned GEMM
- **~26.1 ms mean** (≈ **38.3 Hz**) if all 3 images are present (no fully-masked camera), or if you disable `OPENPI_SKIP_MASKED_IMAGES`
- **~26.8 ms mean** (≈ **37.4 Hz**) without routing fused projections to aiter
- **~31.3 ms mean** (≈ **32.0 Hz**) if you regress back to the slow KV-cache attention path

### What to run (policy inference E2E)

```bash
cd /sgl-workspace/openpi

# Minimal knobs you should set explicitly:
AITER_PRESHUFFLE_WEIGHTS=0 \
OPENPI_SKIP_MASKED_IMAGES=1 \
OPENPI_MANUAL_CUDAGRAPH=1 \
OPENPI_EAGER_ATTN_USE_SDPA=1 \
OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1 \
OPENPI_ROUTE_FUSED_LINEAR_M_THRESH=1000000 \
TORCH_COMPILE_MODE=default \
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
    explicit `bmm + softmax + bmm`. This is the largest single win toward the 23ms perf/W target.
- `OPENPI_SKIP_MASKED_IMAGES=1`
  - If an image is fully masked out for the whole batch (e.g. right wrist camera absent), skip the SigLIP
    vision tower for that image and **drop its image tokens entirely**. This reduces prefix length and KV
    cache length (e.g. 788→532 tokens in the default benchmark) and is the main reason we hit ~22ms.
- `AITER_PRESHUFFLE_WEIGHTS=0`
  - Global bpreshuffle duplicates weights (memory) and can regress small-M decode-ish GEMMs; best-known config keeps it off.
- `OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1` (and a large `..._M_THRESH`)
  - Ensures fused QKV + fused Gate+Up projections use aiter tuned GEMM (and our tuned configs).

### How to profile / confirm

```bash
PROFILE=1 PROFILE_GRAPH_REPLAY=0 PROFILE_DIR=traces/comp_best_repro \
AITER_PRESHUFFLE_WEIGHTS=0 OPENPI_MANUAL_CUDAGRAPH=1 OPENPI_EAGER_ATTN_USE_SDPA=1 \
OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1 OPENPI_ROUTE_FUSED_LINEAR_M_THRESH=1000000 \
TORCH_COMPILE_MODE=default \
python scripts/benchmark_policy_inference.py

python scripts/analyze_policy_trace.py traces/comp_best_repro/policy_inference_compiled_default_call.json
```

