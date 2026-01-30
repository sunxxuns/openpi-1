## MI350 OpenPI policy inference runbook (portable)

This is the **one command** you want on a fresh machine to reproduce the current best end-to-end latency for the Pi0 policy inference benchmark.

### Goal

- **Current best (this repo, MI350)**: ~**31.2 ms** mean for `scripts/benchmark_policy_inference.py` (B1, 10 denoising steps)
- **Perf/Watt target**: **23 ms** (to match H200 perf/watt at 700W vs MI350 1000W)

### What actually worked

- **Keep BF16** (no FP8/INT8 requirements for this path)
- **torch.compile**: use `TORCH_COMPILE_MODE=default` on ROCm
- **Manual full-call CUDAGraph replay**: capture+replay `PI0Pytorch.sample_actions(...)`
  - ROCm capture can fail if Dynamo traces during capture because Dynamo saves CUDA RNG state.
  - We patch Dynamo at runtime (best-effort) to skip CUDA RNG get_state while a stream capture is active.
- **GEMM**: route `nn.Linear` to **aiter tuned GEMM** dispatcher; optionally **pre-shuffle** eligible Linear weights to enable **bpreshuffle asm kernels** on gfx950.
- **Attention (important)**: compile *through* aiter attention with the direct `mha_fwd` path.
  - `OPENPI_DISABLE_COMPILE_AITER_ATTN=0`
  - `OPENPI_AITER_ATTN_DIRECT_MHA=1`
  - `OPENPI_INDUCTOR_MEMORY_PLANNING=0` (Inductor stability on ROCm; also reduces kernel count)

### Recommended environment

```bash
export HIP_LAUNCH_BLOCKING=0
export AMD_LOG_LEVEL=0
export HIP_CACHE_ENABLED=1
```

### Best known command (end-to-end)

```bash
cd /sgl-workspace/openpi

AITER_PRESHUFFLE_WEIGHTS=1 \
OPENPI_MANUAL_CUDAGRAPH=1 \
TORCH_COMPILE_MODE=default \
python scripts/benchmark_policy_inference.py
```

Expected: **~31.2 ms** mean, **~32 Hz** throughput (on MI350).

### Notes / caveats

- **Memory**: `AITER_PRESHUFFLE_WEIGHTS=1` keeps both original + shuffled weights, so peak memory is higher.
- **First run**: aiter may JIT/build some modules on first execution; benchmark uses warmup, but overall script runtime may be longer the first time.
- **If capture fails**: the script prints a warning and falls back to normal execution. If that happens, rerun with a larger warmup, or disable manual graph and use compiled-only mode.
- **If you see ~34-35ms again**: you likely re-enabled one of these (unset them):
  - `OPENPI_DISABLE_COMPILE_AITER_ATTN=1` (graph-breaks around attention; increases kernel count)
  - `OPENPI_INDUCTOR_MEMORY_PLANNING=1` (can trigger Inductor issues on ROCm / regresses kernel graph)

