## MI350 OpenPI policy inference runbook (portable)

This is the **one command** you want on a fresh machine to reproduce the current best end-to-end latency for the Pi0 policy inference benchmark.

### Goal

- **Current best (this repo, MI350)**: ~**22.2 ms** mean for `scripts/benchmark_policy_inference.py` (B1, 10 denoising steps)
- **Perf/Watt target**: **23 ms** (to match H200 perf/watt at 700W vs MI350 1000W)

### What actually worked

- **Keep BF16** (no FP8/INT8 requirements for this path)
- **Skip fully-masked cameras** (big win for the default benchmark):
  - `OPENPI_SKIP_MASKED_IMAGES=1` drops image tokens for cameras that are fully masked out.
- **torch.compile**: use `TORCH_COMPILE_MODE=default` on ROCm
- **Manual full-call CUDAGraph replay**: capture+replay `PI0Pytorch.sample_actions(...)`
  - ROCm capture can fail if Dynamo traces during capture because Dynamo saves CUDA RNG state.
  - We patch Dynamo at runtime (best-effort) to skip CUDA RNG get_state while a stream capture is active.
- **GEMM**:
  - route `nn.Linear` to **aiter tuned GEMM** dispatcher
  - route **fused QKV + fused Gate+Up** projections through aiter tuned GEMM (so they use our tuned configs too)
  - keep **global bpreshuffle off** (`AITER_PRESHUFFLE_WEIGHTS=0`) for best end-to-end latency
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
OPENPI_SKIP_MASKED_IMAGES=1 \
OPENPI_MANUAL_CUDAGRAPH=1 \
OPENPI_EAGER_ATTN_USE_SDPA=1 \
OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1 \
OPENPI_ROUTE_FUSED_LINEAR_M_THRESH=1000000 \
TORCH_COMPILE_MODE=default \
python scripts/benchmark_policy_inference.py
```

Expected: **~22.2 ms** mean, **~45 Hz** throughput (on MI350), assuming at least one camera is fully masked (default benchmark).
If all 3 images are present (no fully-masked camera), expect closer to **~26 ms**.

### Notes / caveats

- **Memory**: `AITER_PRESHUFFLE_WEIGHTS=0` keeps memory lower and avoids regressions for small-M GEMMs.
- **First run**: aiter may JIT/build some modules on first execution; benchmark uses warmup, but overall script runtime may be longer the first time.
- **If capture fails**: the script prints a warning and falls back to normal execution. If that happens, rerun with a larger warmup, or disable manual graph and use compiled-only mode.
- **If you see ~34-35ms again**: you likely re-enabled one of these (unset them):
  - `OPENPI_DISABLE_COMPILE_AITER_ATTN=1` (graph-breaks around attention; increases kernel count)
  - `OPENPI_INDUCTOR_MEMORY_PLANNING=1` (can trigger Inductor issues on ROCm / regresses kernel graph)

