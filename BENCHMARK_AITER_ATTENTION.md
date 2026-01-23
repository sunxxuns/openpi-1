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

## Precision Verification

```bash
python scripts/verify_precision.py
```

| Metric | Value |
|--------|-------|
| Cosine Similarity | **1.000000** |
| Result | **PASSED** |
