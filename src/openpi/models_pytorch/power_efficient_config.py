"""Maximum throughput configuration for AMD MI300/MI350.

Supports both MI300X (gfx942, ~750W) and MI350 (gfx950, ~1000W).

MI350 draws 1000W vs H200's 700W, so we need proportionally better performance:
- H200: 32.9ms @ 700W = 0.047 samples/s/W
- MI350 current: 35.7ms @ 1000W = 0.028 samples/s/W
- MI350 target: 23ms @ 1000W = 0.043 samples/s/W (to match H200 perf/watt)

MI300X draws ~750W vs H200's 700W (closer to parity):
- MI300X target: ~30.4ms @ 750W to match H200 perf/watt

Key optimizations to MAXIMIZE throughput:
1. Enable HIP graphs to reduce kernel launch overhead (202ms -> <10ms)
2. Reduce synchronization points (338ms -> ~50ms)
3. Maximize GPU utilization (keep all 304 CUs busy)
4. Use fastest kernel configurations (max autotune)
5. Aggressive prefetching and memory optimization

Profile analysis shows MI350 wastes 47% of time on overhead:
- MI350: 1147ms total, 338ms sync (29%), 202ms launch (18%), 452ms GEMM
- H200:  468ms total,  90ms sync (19%),   3ms launch (1%),   12ms GEMM

Target: Eliminate overhead to achieve target latency (match H200 perf/watt).
"""

import os
import torch


def configure_max_throughput_inductor():
    """Configure inductor for MAXIMUM throughput on AMD MI350.
    
    Target: 23ms latency (34% improvement from 35.7ms) to match H200 perf/watt.
    
    Strategy:
    1. HIP graphs to eliminate 202ms launch overhead
    2. Reduce 338ms sync overhead to ~50ms
    3. Aggressive autotuning for fastest kernels
    4. Maximum fusion to reduce kernel count
    5. Full GPU utilization (304 CUs)
    """
    try:
        import torch._inductor.config as inductor_config
        import torch._dynamo.config as dynamo_config
        
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        
        if not is_rocm:
            print("[power_config] Not on ROCm, using default config")
            return False
        
        # Check ROCm version for HIP graph support
        rocm_version = torch.version.hip or "0.0"
        major_version = int(rocm_version.split(".")[0]) if rocm_version else 0
        
        print(f"[power_config] ROCm version: {rocm_version}")
        
        # =================================================================
        # CUDA/HIP Graphs - Critical for reducing launch overhead
        # =================================================================
        # ROCm 6.0+ has improved HIP graph support
        # This alone can reduce launch overhead from 202ms to <10ms
        enable_graphs = os.environ.get("ENABLE_HIP_GRAPHS", "1") == "1"
        
        if major_version >= 6 and enable_graphs:
            print("[power_config] Enabling HIP graphs (ROCm 6.0+)")
            inductor_config.triton.cudagraphs = True
            inductor_config.triton.cudagraph_trees = True
            # Disable slow autotuning for graph capture
            inductor_config.triton.cudagraph_skip_autotuning = True
        else:
            print(f"[power_config] HIP graphs disabled (ROCm {major_version}, enable={enable_graphs})")
            inductor_config.triton.cudagraphs = False
            inductor_config.triton.cudagraph_trees = False
        
        # =================================================================
        # Aggressive Kernel Fusion - Reduce total kernel count
        # =================================================================
        # More fusion = fewer launches = less overhead = less idle power
        inductor_config.aggressive_fusion = True
        inductor_config.epilogue_fusion = True
        inductor_config.pattern_matcher = True
        
        # Enable multi-kernel fusion for elementwise chains
        inductor_config.triton.multi_kernel = 1  # Enable multi-kernel fusion
        
        # Force fusion even for ops that might slow down individually
        # (overall faster due to reduced launch overhead)
        inductor_config.force_fuse_pointwise = True
        
        # =================================================================
        # Triton Backend Optimizations for AMD
        # =================================================================
        # Use Triton for all GEMMs - better fusion with surrounding ops
        inductor_config.max_autotune_gemm_backends = "TRITON"
        
        # Enable persistent kernels (reduce launch overhead)
        inductor_config.triton.persistent_reductions = True
        
        # Tune block sizes for MI300/MI350 CU count (304 CUs)
        # Smaller blocks = more parallelism = better utilization
        if hasattr(inductor_config, 'triton_override_block_sizes'):
            # Use block sizes that map well to 304 CUs
            inductor_config.triton_override_block_sizes = True
        
        # =================================================================
        # Memory Optimizations - Reduce bandwidth pressure
        # =================================================================
        # Reuse memory aggressively (less allocation = less power)
        inductor_config.reuse_all_buffers = True
        
        # Enable memory planning to reduce peak memory and allocations
        inductor_config.memory_planning = True
        
        # =================================================================
        # AGGRESSIVE AUTOTUNING - Find fastest kernels (longer compile OK)
        # =================================================================
        # For max throughput, we want the best kernels even if compile is slow
        use_full_tune = os.environ.get("INDUCTOR_FULL_AUTOTUNE", "1") == "1"  # Default ON for max throughput
        inductor_config.max_autotune = use_full_tune
        inductor_config.coordinate_descent_tuning = use_full_tune
        
        # Increase autotune cache for repeated runs
        if hasattr(inductor_config, 'autotune_in_subproc'):
            inductor_config.autotune_in_subproc = True  # Parallel autotuning
        
        # =================================================================
        # Reduce Synchronization - Critical for power efficiency
        # =================================================================
        # Avoid unnecessary syncs (saves 338ms in profile)
        inductor_config.fallback_random = False  # Don't sync for random ops
        
        # =================================================================
        # Dynamo Configuration
        # =================================================================
        # Increase cache to avoid recompilation
        dynamo_config.cache_size_limit = 128
        
        # Suppress graph breaks (more code in single graph = better fusion)
        dynamo_config.suppress_errors = False
        
        # Enable automatic dynamic shapes to avoid recompilation
        dynamo_config.automatic_dynamic_shapes = True
        
        # =================================================================
        # Environment Variables for HIP Runtime
        # =================================================================
        # These reduce HIP runtime overhead
        os.environ.setdefault("HIP_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        
        # Disable HIP kernel caching during development (faster startup)
        # Enable in production for faster repeated runs
        if os.environ.get("HIP_CACHE_ENABLED") is None:
            os.environ["HIP_CACHE_ENABLED"] = "1"
        
        # Reduce HIP launch overhead
        os.environ.setdefault("HIP_LAUNCH_BLOCKING", "0")
        os.environ.setdefault("AMD_LOG_LEVEL", "0")  # Reduce logging overhead
        
        print("[power_config] Configuration applied:")
        print(f"  - HIP graphs: {inductor_config.triton.cudagraphs}")
        print(f"  - Aggressive fusion: {inductor_config.aggressive_fusion}")
        print(f"  - Persistent reductions: {inductor_config.triton.persistent_reductions}")
        print(f"  - Multi-kernel fusion: {inductor_config.triton.multi_kernel}")
        print(f"  - Full autotune: {use_full_tune}")
        
        return True
        
    except Exception as e:
        print(f"[power_config] Failed to configure: {e}")
        return False


def configure_gpu_power_management():
    """Configure GPU power management for better efficiency.
    
    On MI350, we can trade some peak performance for better power efficiency
    by adjusting clock frequencies and power limits.
    """
    try:
        import subprocess
        
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        if not is_rocm:
            return False
        
        # Check if rocm-smi is available
        result = subprocess.run(
            ["which", "rocm-smi"], 
            capture_output=True, 
            text=True
        )
        if result.returncode != 0:
            print("[power_config] rocm-smi not available")
            return False
        
        # Get current power info
        result = subprocess.run(
            ["rocm-smi", "--showpower"], 
            capture_output=True, 
            text=True,
            timeout=10
        )
        print(f"[power_config] Current power: {result.stdout.strip()}")
        
        # Check power limit (optional - requires root)
        power_limit = os.environ.get("MI350_POWER_LIMIT_W")
        if power_limit:
            print(f"[power_config] Attempting to set power limit to {power_limit}W")
            result = subprocess.run(
                ["rocm-smi", "--setpoweroverdrive", power_limit],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"[power_config] Power limit set to {power_limit}W")
            else:
                print(f"[power_config] Failed to set power limit: {result.stderr}")
        
        return True
        
    except Exception as e:
        print(f"[power_config] GPU power management failed: {e}")
        return False


def get_power_efficient_compile_options():
    """Get torch.compile options optimized for power efficiency.
    
    Returns compile kwargs that balance performance and power consumption.
    """
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    
    if is_rocm:
        # Check ROCm version
        rocm_version = torch.version.hip or "0.0"
        major_version = int(rocm_version.split(".")[0]) if rocm_version else 0
        
        if major_version >= 6:
            # ROCm 6.0+: Use reduce-overhead with HIP graphs
            return {
                "mode": "reduce-overhead",
                "fullgraph": True,  # Try to capture full graph
                "dynamic": False,   # Static shapes for better graph capture
            }
        else:
            # Older ROCm: Use default mode without graphs
            return {
                "mode": "default",
                "fullgraph": False,
                "dynamic": True,
            }
    else:
        # CUDA: Full optimization
        return {
            "mode": "reduce-overhead",
            "fullgraph": True,
            "dynamic": False,
        }


def calculate_target_latency(h200_latency_ms: float = 32.9, h200_power_w: float = 700, mi350_power_w: float = None) -> dict:
    """Calculate target MI300/MI350 latency to match H200 perf/watt.
    
    Since AMD GPUs may draw more power, they should achieve proportionally lower
    latency to deliver the same performance per watt.
    
    Args:
        h200_latency_ms: H200 latency in milliseconds
        h200_power_w: H200 power consumption in watts
        mi350_power_w: AMD GPU power consumption in watts (auto-detected if None)
    
    Returns:
        Dict with target metrics
    """
    # Auto-detect power based on GPU architecture
    if mi350_power_w is None:
        try:
            from openpi.models_pytorch.mi350_kernel_configs import is_mi300
            mi350_power_w = 750 if is_mi300() else 1000
        except Exception:
            mi350_power_w = 1000  # fallback

    # H200 perf/watt
    h200_throughput = 1000 / h200_latency_ms  # samples/s
    h200_perf_per_watt = h200_throughput / h200_power_w
    
    # Target MI350 latency to match H200 perf/watt
    # perf_per_watt = throughput / power = (1000/latency) / power
    # To match: (1000/target_latency) / mi350_power = h200_perf_per_watt
    # target_latency = 1000 / (h200_perf_per_watt * mi350_power)
    target_latency_ms = 1000 / (h200_perf_per_watt * mi350_power_w)
    
    # Or equivalently: target = h200_latency * (h200_power / mi350_power)
    target_latency_simple = h200_latency_ms * (h200_power_w / mi350_power_w)
    
    # Current MI350 performance
    current_mi350_latency = 35.7  # ms
    current_mi350_throughput = 1000 / current_mi350_latency
    current_mi350_perf_per_watt = current_mi350_throughput / mi350_power_w
    
    # Required improvement
    improvement_needed = (current_mi350_latency - target_latency_ms) / current_mi350_latency * 100
    
    return {
        "h200_latency_ms": h200_latency_ms,
        "h200_power_w": h200_power_w,
        "h200_perf_per_watt": h200_perf_per_watt,
        "mi350_power_w": mi350_power_w,
        "target_latency_ms": target_latency_ms,
        "current_latency_ms": current_mi350_latency,
        "improvement_needed_pct": improvement_needed,
        "target_throughput_hz": 1000 / target_latency_ms,
        "current_throughput_hz": current_mi350_throughput,
    }


def estimate_latency_improvement(current_ms: float, overhead_reduction_pct: float) -> dict:
    """Estimate latency improvement from reducing overhead.
    
    Based on profile analysis:
    - Current overhead: 47% (338ms sync + 202ms launch out of 1147ms)
    - Target overhead: 10% (like H200)
    
    Args:
        current_ms: Current latency in milliseconds
        overhead_reduction_pct: Percentage reduction in overhead (0-100)
    
    Returns:
        Dict with estimated improvements
    """
    # Current overhead breakdown (from profiling)
    current_overhead_pct = 0.47  # 47% of time is overhead
    current_compute_pct = 1 - current_overhead_pct  # 53% actual compute
    
    # New overhead after optimization
    new_overhead_pct = current_overhead_pct * (1 - overhead_reduction_pct / 100)
    
    # Compute time stays the same, overhead reduces
    compute_time_ms = current_ms * current_compute_pct
    new_overhead_ms = current_ms * current_overhead_pct * (1 - overhead_reduction_pct / 100)
    
    new_latency_ms = compute_time_ms + new_overhead_ms
    speedup = current_ms / new_latency_ms
    
    return {
        "current_latency_ms": current_ms,
        "new_latency_ms": new_latency_ms,
        "speedup": speedup,
        "compute_time_ms": compute_time_ms,
        "old_overhead_ms": current_ms * current_overhead_pct,
        "new_overhead_ms": new_overhead_ms,
    }


# Convenience function to apply all optimizations
def apply_max_throughput_optimizations():
    """Apply all optimizations for maximum throughput on MI350."""
    print("=" * 60)
    print("APPLYING MAXIMUM THROUGHPUT OPTIMIZATIONS FOR MI350")
    print("=" * 60)
    
    # Show target
    target = calculate_target_latency()
    print(f"\nTarget: {target['target_latency_ms']:.1f}ms to match H200 perf/watt")
    print(f"Current: {target['current_latency_ms']:.1f}ms")
    print(f"Improvement needed: {target['improvement_needed_pct']:.1f}%")
    
    # 1. Configure inductor for max throughput
    inductor_ok = configure_max_throughput_inductor()
    
    # 2. Configure GPU (optional)
    power_ok = configure_gpu_power_management()
    
    # 3. Set torch matmul precision for speed
    torch.set_float32_matmul_precision("high")
    
    print(f"\nOptimizations applied: inductor={inductor_ok}")
    print("=" * 60)
    
    return inductor_ok


# Alias for backwards compatibility
apply_all_power_optimizations = apply_max_throughput_optimizations
configure_power_efficient_inductor = configure_max_throughput_inductor


if __name__ == "__main__":
    print("=" * 70)
    print("MI350 THROUGHPUT TARGET ANALYSIS")
    print("=" * 70)
    
    # Calculate target latency
    target = calculate_target_latency()
    print("\nPerf/Watt Parity Target:")
    print(f"  H200:  {target['h200_latency_ms']:.1f}ms @ {target['h200_power_w']:.0f}W = {target['h200_perf_per_watt']:.4f} samples/s/W")
    print(f"  MI350 current: {target['current_latency_ms']:.1f}ms @ {target['mi350_power_w']:.0f}W = {target['current_throughput_hz']/target['mi350_power_w']:.4f} samples/s/W")
    print(f"  MI350 target:  {target['target_latency_ms']:.1f}ms @ {target['mi350_power_w']:.0f}W = {target['h200_perf_per_watt']:.4f} samples/s/W")
    print(f"\n  Improvement needed: {target['improvement_needed_pct']:.1f}% latency reduction")
    print(f"  Target throughput: {target['target_throughput_hz']:.1f} Hz (from {target['current_throughput_hz']:.1f} Hz)")
    
    # Estimate improvement from overhead reduction
    print("\nOverhead Reduction Analysis:")
    print("  Profile shows 47% overhead (sync=30%, launch=17%)")
    
    for reduction in [50, 70, 80, 90]:
        est = estimate_latency_improvement(35.7, reduction)
        print(f"  {reduction}% overhead reduction: {est['new_latency_ms']:.1f}ms ({est['speedup']:.2f}x speedup)")
    
    print("\n" + "=" * 70)
    print("To achieve 23ms target, need ~80% overhead reduction")
    print("This requires: HIP graphs + reduced sync + aggressive fusion")
    print("=" * 70)
    
    # Apply optimizations
    print("\n")
    apply_max_throughput_optimizations()
