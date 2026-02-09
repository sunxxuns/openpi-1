#!/bin/bash
# Batch size sweep for MI300 benchmark (no skip masked images, apples-to-apples with H200)
set -euo pipefail

export AITER_PRESHUFFLE_WEIGHTS=0
export OPENPI_SKIP_MASKED_IMAGES=0
export OPENPI_MANUAL_CUDAGRAPH=1
export OPENPI_EAGER_ATTN_USE_SDPA=1
export OPENPI_FUSE_SIGLIP_QKV=1
export OPENPI_ROUTE_SIGLIP_FUSED_QKV_TO_AITER=1
export OPENPI_ROUTE_FUSED_LINEAR_TO_AITER=1
export OPENPI_ROUTE_FUSED_LINEAR_M_THRESH=1000000
export TORCH_COMPILE_MODE=default

RESULTS_FILE="$(dirname "$0")/../mi300_batch_sweep_results.txt"
echo "MI300 Batch Size Sweep Results" > "$RESULTS_FILE"
echo "==============================" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

for BSZ in 2 4 8 16 32 64; do
    echo ""
    echo "============================================"
    echo "  BATCH SIZE = $BSZ"
    echo "============================================"
    echo ""
    python scripts/benchmark_policy_inference.py --batch-size "$BSZ" --warmup 10 --iterations 20 2>&1 | tee /tmp/bsz_${BSZ}_output.txt

    # Extract results
    MEAN=$(grep "Mean latency:" /tmp/bsz_${BSZ}_output.txt | awk '{print $3}')
    P50=$(grep "P50:" /tmp/bsz_${BSZ}_output.txt | awk '{print $2}')
    HZ=$(grep "Throughput:" /tmp/bsz_${BSZ}_output.txt | awk '{print $2}')
    SAMPLES_LINE=$(grep "Samples/s:" /tmp/bsz_${BSZ}_output.txt || true)

    echo "BSZ=$BSZ  Mean=${MEAN}ms  P50=${P50}ms  Hz=${HZ}" >> "$RESULTS_FILE"
    if [ -n "$SAMPLES_LINE" ]; then
        SAMPLES=$(echo "$SAMPLES_LINE" | awk '{print $2}')
        echo "  Samples/s=$SAMPLES" >> "$RESULTS_FILE"
    fi
done

echo "" >> "$RESULTS_FILE"
echo "Done." >> "$RESULTS_FILE"
echo ""
echo "==============================="
echo "  SWEEP COMPLETE"
echo "==============================="
cat "$RESULTS_FILE"
