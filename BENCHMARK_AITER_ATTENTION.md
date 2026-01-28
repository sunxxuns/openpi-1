# AMD MI350 Benchmark - OpenPI Pi0 (3.5B)

## Full Policy Inference (batch=1)

```bash
python scripts/benchmark_policy_inference.py
```

| Metric | Value |
|--------|-------|
| Latency | **137.3 ms** |
| Throughput | **7.28 Hz** |
| Memory | 7.10 GB |

## Full Policy Inference (batch=1, torch.compile)

Benchmark on gfx950 (MI355X) with fused projections + aiter GEMM enabled.
Mask override is used to skip expensive mask classification in aiter attention (benchmark-only).

```bash
TORCH_COMPILE_MODE=reduce-overhead \
USE_FUSED_PROJECTIONS=1 USE_AITER_GEMM=1 USE_OPTIMIZED_OPS=1 \
AITER_MASK_OVERRIDE=1 \
python scripts/benchmark_policy_inference.py
```

| Metric | Value |
|--------|-------|
| Mean latency | **35.7 ms** |
| Throughput | **28.04 Hz** |
| Memory | 10.03 GB |

## Precision Verification

```bash
python scripts/verify_precision.py
```

| Metric | Value |
|--------|-------|
| Cosine Similarity | **1.000000** |
| Result | **PASSED** |
