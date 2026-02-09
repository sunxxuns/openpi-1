#!/bin/bash
# Sweep all batch sizes and collect results
set -e

export HIP_LAUNCH_BLOCKING=0
export AMD_LOG_LEVEL=0
export HIP_CACHE_ENABLED=1
export AITER_PRESHUFFLE_WEIGHTS=0
export OPENPI_SKIP_MASKED_IMAGES=0
export OPENPI_MANUAL_CUDAGRAPH=1
export OPENPI_EAGER_ATTN_USE_SDPA=1
export OPENPI_FUSE_SIGLIP_QKV=1
export OPENPI_ROUTE_SIGLIP_FUSED_QKV_TO_AITER=1
export OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1
export OPENPI_ROUTE_FUSED_LINEAR_M_THRESH=1000000
export TORCH_COMPILE_MODE=default
export OPENPI_BATCH_SIGLIP=1
export OPENPI_COORD_DESCENT=1
export OPENPI_BENCHMARK_KERNEL=1
export OPENPI_GROUP_FUSION=1
export OPENPI_MAX_FUSION_SIZE=128
export OPENPI_FREEZING=1

cd /sgl-workspace/openpi

echo "BSZ,Mean_ms,P50_ms,P95_ms,Hz,Samples_per_s"

for BSZ in 1 2 4 8 16 32 64; do
    echo "--- Running BSZ=$BSZ ---" >&2
    result=$(OPENPI_BATCH_SIZE=$BSZ python scripts/benchmark_policy_inference.py \
        --batch-size $BSZ --warmup 10 --iterations 30 2>&1 | \
        python3 -c "
import sys
lines = sys.stdin.readlines()
mean = p50 = p95 = hz = ''
for l in lines:
    l = l.strip()
    if l.startswith('Mean latency:'):   mean = l.split()[-2]
    if l.startswith('P50:'):            p50 = l.split()[-2]
    if l.startswith('P95:'):            p95 = l.split()[-2]
    if l.startswith('Throughput:'):      hz = l.split()[-2]
    if l.startswith('Samples/s:'):      pass
bsz = $BSZ
hz_f = float(hz)
sps = hz_f * bsz
print(f'{bsz},{mean},{p50},{p95},{hz},{sps:.1f}')
")
    echo "$result"
done
