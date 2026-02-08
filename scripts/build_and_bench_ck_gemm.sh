#!/bin/bash
# Build and benchmark CK GEMM kernels for M=788 shapes on gfx950
set -e

CK_DIR=/sgl-workspace/aiter/3rdparty/composable_kernel
BUILD_DIR=/tmp/ck_gemm_build
BENCH_SRC=$CK_DIR/tile_engine/ops/gemm/gemm_benchmark_single.cpp
CK_INC=$CK_DIR/include

export HIP_VISIBLE_DEVICES=7

mkdir -p $BUILD_DIR

# Promising tile configs for M=788: tile_m in {64,128}, various N/K tiles
# Format: TileM x TileN x TileK _ WarpM x WarpN x WarpK _ WarpTileM x WarpTileN x WarpTileK
CONFIGS=(
    "128x128x64_2x2x1_32x32x16"
    "128x128x64_2x2x1_16x16x32"
    "128x128x64_4x1x1_32x32x16"
    "128x128x64_1x4x1_32x32x16"
    "128x64x64_2x2x1_32x32x16"
    "128x64x64_4x1x1_32x32x16"
    "64x128x64_2x2x1_32x32x16"
    "64x128x64_1x4x1_32x32x16"
    "64x64x128_2x2x1_32x32x16"
    "64x64x128_4x1x1_16x16x32"
    "192x64x64_2x2x1_32x32x16"
)

TRAIT="compv4_cshuffle_intrawave_False_False_False_False"

echo "=== Building CK GEMM kernels for gfx950 ==="
echo "Configs: ${#CONFIGS[@]}"

for cfg in "${CONFIGS[@]}"; do
    KNAME="gemm_bf16_rcr_${TRAIT}_${cfg}"
    HEADER_PATH="$BUILD_DIR/${KNAME}.hpp"

    # Generate kernel header
    python $CK_DIR/tile_engine/ops/gemm/gemm_instance_builder.py \
        --working_path $BUILD_DIR \
        --gpu_target gfx950 \
        --datatype bf16 \
        --layout rcr \
        --config_json $CK_DIR/tile_engine/ops/gemm/configs/default_config.json \
        --gen_single \
        --kernel_name "$KNAME" \
        --tile_config "$cfg" \
        --trait_combo "$TRAIT" 2>/dev/null

    if [ ! -f "$BUILD_DIR/gemm_single_${KNAME}.hpp" ]; then
        echo "SKIP $cfg (header generation failed)"
        continue
    fi

    # Compile
    EXE="$BUILD_DIR/bench_${cfg}"
    echo -n "Compiling $cfg... "
    hipcc -x hip --offload-arch=gfx950 \
        -I$CK_INC \
        -Wno-undefined-func-template -Wno-float-equal -Wno-gnu-line-marker \
        --offload-compress -O3 \
        -DGEMM_SINGLE_HEADER="\"gemm_single_${KNAME}.hpp\"" \
        -include "$BUILD_DIR/gemm_single_${KNAME}.hpp" \
        $BENCH_SRC \
        -o "$EXE" 2>/dev/null

    if [ $? -ne 0 ]; then
        echo "FAILED"
        continue
    fi
    echo "OK"

    # Benchmark for our hot shapes
    for shape in "788,2048,2048" "788,2560,2048" "788,2048,16384" "788,32768,2048"; do
        IFS=',' read -r M N K <<< "$shape"
        result=$($EXE -m $M -n $N -k $K -w 50 -r 200 2>/dev/null | grep "avg" || echo "FAIL")
        echo "  M=$M N=$N K=$K: $result"
    done
done

echo ""
echo "=== Reference: torch.mm (rocBLAS) ==="
python3 -c "
import torch, os
os.environ['AMD_LOG_LEVEL']='0'
dev = torch.device('cuda:0')
torch.cuda.set_device(dev)
for M,N,K in [(788,2048,2048),(788,2560,2048),(788,2048,16384),(788,32768,2048)]:
    a=torch.randn(M,K,dtype=torch.bfloat16,device=dev)
    b=torch.randn(K,N,dtype=torch.bfloat16,device=dev)
    for _ in range(50): torch.mm(a,b)
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(200): torch.mm(a,b)
    e.record(); e.synchronize()
    us=s.elapsed_time(e)*1000/200
    tflops=2*M*N*K/(us*1e-6)/1e12
    print(f'  M={M} N={N} K={K}: {us:.1f} us ({tflops:.0f} TFLOPS)')
"
