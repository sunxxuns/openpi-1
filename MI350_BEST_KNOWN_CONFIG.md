## MI350 Best Known Config (do not regress)

This file exists to prevent “we forgot the magic env vars” regressions.

### What to run (policy inference E2E)

```bash
cd /sgl-workspace/openpi

# Minimal knobs you should set explicitly:
AITER_PRESHUFFLE_WEIGHTS=1 \
OPENPI_MANUAL_CUDAGRAPH=1 \
TORCH_COMPILE_MODE=default \
python scripts/benchmark_policy_inference.py
```

### The 31ms-critical invariants

These must remain true (either by script defaults or by explicit env vars):

- `OPENPI_DISABLE_COMPILE_AITER_ATTN=0`
  - If set to `1`, we graph-break around aiter attention and regress to ~34ms due to more kernels.
- `OPENPI_AITER_ATTN_DIRECT_MHA=1`
  - Forces the direct `aiter.ops.mha.mha_fwd` call (more `torch.compile` friendly).
- `OPENPI_INDUCTOR_MEMORY_PLANNING=0`
  - Avoids Inductor ROCm recursion issues and keeps the compiled graph/kernel-count low.

### How to profile / confirm

```bash
PROFILE=1 PROFILE_GRAPH_REPLAY=1 PROFILE_DIR=traces/comp_31_repro \
AITER_PRESHUFFLE_WEIGHTS=1 OPENPI_MANUAL_CUDAGRAPH=1 TORCH_COMPILE_MODE=default \
python scripts/benchmark_policy_inference.py

python scripts/analyze_policy_trace.py traces/comp_31_repro/policy_inference_compiled_default_replay.json
```

