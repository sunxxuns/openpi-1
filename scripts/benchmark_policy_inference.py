#!/usr/bin/env python3
"""
Benchmark full Pi0 policy inference (batch_size=1).

Tests the complete inference pipeline:
- Image encoding (SigLIP vision tower)
- Language encoding (Gemma embedding)
- Prefill (KV cache generation for images + text)
- Denoising steps (10 steps of action generation)

This measures realistic end-to-end latency for robot control.

IMPORTANT: This benchmark tests the public model API (model.sample_actions).
All backend optimizations (custom attention kernels, GEMM tuning, CUDAGraph
capture, torch.compile settings, etc.) should be implemented INSIDE the model
code or via environment configuration — NOT by modifying this script.

Usage:
    # Basic benchmark (wall-clock timing)
    python scripts/benchmark_policy_inference.py

    # CUDA event timing (measures GPU time, avoids sync overhead)
    python scripts/benchmark_policy_inference.py --timing cuda_event

    # With CUDAGraph capture+replay (reduces CPU launch overhead)
    python scripts/benchmark_policy_inference.py --cudagraph

    # With profiling
    python scripts/benchmark_policy_inference.py --profile
"""

import os
import sys
import pathlib
import argparse
import time

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

# Make repo `src/` importable when running from a source checkout.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Benchmark Pi0 policy inference")
    p.add_argument(
        "--gpu", type=int,
        default=int(os.environ.get("OPENPI_GPU_ID", "0")),
        help="GPU device id (default: OPENPI_GPU_ID or 0)",
    )
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size (default: 1)",
    )
    p.add_argument(
        "--warmup", type=int, default=10,
        help="Warmup iterations (default: 10)",
    )
    p.add_argument(
        "--iterations", type=int, default=30,
        help="Benchmark iterations (default: 30)",
    )
    p.add_argument(
        "--num-steps", type=int, default=10,
        help="Denoising steps (default: 10)",
    )
    p.add_argument(
        "--timing", choices=["wall", "cuda_event"], default="wall",
        help="Timing method: wall or cuda_event (default: wall)",
    )
    p.add_argument(
        "--cudagraph", action="store_true",
        help="Enable manual CUDAGraph capture+replay for lower CPU overhead",
    )
    p.add_argument(
        "--profile", action="store_true",
        help="Enable PyTorch profiler and export chrome trace",
    )
    p.add_argument(
        "--profile-dir", default="traces",
        help="Directory for profile traces (default: traces)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("PI0 FULL POLICY INFERENCE BENCHMARK")
    print("=" * 70)

    gpu_id = args.gpu
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    print(f"Device: {torch.cuda.get_device_name(gpu_id)} (cuda:{gpu_id})")
    print(f"PyTorch: {torch.__version__}")

    # ------------------------------------------------------------------ #
    # Create model
    # ------------------------------------------------------------------ #
    print("\nCreating Pi0 model...")

    from dataclasses import dataclass
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    @dataclass
    class Pi0ConfigPytorch:
        action_dim: int = 32
        action_horizon: int = 10
        max_token_len: int = 48
        dtype: str = "bfloat16"
        paligemma_variant: str = "gemma_2b"
        action_expert_variant: str = "gemma_300m"
        pi05: bool = False

    config = Pi0ConfigPytorch()
    model = PI0Pytorch(config)
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model parameters: {param_count:.2f}B")

    # ------------------------------------------------------------------ #
    # Create synthetic observation
    # ------------------------------------------------------------------ #
    batch_size = args.batch_size
    print(f"\nCreating observation (batch_size={batch_size})...")

    class SimpleObservation:
        """Lightweight observation container matching the model's expected interface."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    images = {
        "base_0_rgb": torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device) * 2 - 1,
        "left_wrist_0_rgb": torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device) * 2 - 1,
        "right_wrist_0_rgb": torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
    }
    image_masks = {
        "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool, device=device),
    }
    state = torch.randn(batch_size, 32, dtype=torch.bfloat16, device=device)
    tokenized_prompt = torch.randint(0, 256000, (batch_size, 20), dtype=torch.long, device=device)
    tokenized_prompt_mask = torch.ones(batch_size, 20, dtype=torch.bool, device=device)

    observation = SimpleObservation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _sync():
        """GPU synchronization (works on both CUDA and ROCm)."""
        torch.cuda.current_stream().synchronize()

    num_steps = args.num_steps
    warmup = args.warmup
    iterations = args.iterations

    print(f"Denoising steps: {num_steps}")
    print(f"Warmup: {warmup}, Iterations: {iterations}")
    print("-" * 50)

    # ------------------------------------------------------------------ #
    # Warmup
    # ------------------------------------------------------------------ #
    print("Warmup...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=num_steps)
    _sync()

    # ------------------------------------------------------------------ #
    # Optional: manual CUDAGraph capture+replay
    # ------------------------------------------------------------------ #
    graph = None
    static_actions = None
    if args.cudagraph:
        print("\nManual CUDAGraph capture...")
        try:
            pool = torch.cuda.graphs.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()

            # Extra warmup with the exact callable we will capture.
            with torch.no_grad():
                for _ in range(5):
                    _ = model.sample_actions(device, observation, num_steps=num_steps)
            _sync()

            print("Capturing graph (1 iteration)...")
            with torch.cuda.graph(graph, pool=pool):
                static_actions = model.sample_actions(
                    device, observation, num_steps=num_steps
                )
            _sync()
            print("Graph capture succeeded.")
        except Exception as e:
            print(f"Warning: CUDAGraph capture failed, falling back to normal: {e}")
            graph = None
            static_actions = None

    # ------------------------------------------------------------------ #
    # Optional: profiling
    # ------------------------------------------------------------------ #
    if args.profile:
        trace_dir = args.profile_dir
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, "policy_inference_benchmark.json")
        print("\nProfiling (1 iteration)...")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                if graph is not None:
                    graph.replay()
                else:
                    _ = model.sample_actions(device, observation, num_steps=num_steps)
            _sync()
        prof.export_chrome_trace(trace_path)
        trace_size_mb = os.path.getsize(trace_path) / 1024 / 1024
        print(f"Trace saved: {trace_path} ({trace_size_mb:.1f} MB)")
        print("\nTop ops by self CUDA time:")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

    # ------------------------------------------------------------------ #
    # Benchmark loop
    # ------------------------------------------------------------------ #
    timing = args.timing.lower()
    print(f"\nBenchmarking... (timing={timing})")
    latencies = []

    use_events = timing == "cuda_event"
    ev_start = ev_end = None
    if use_events:
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)

    for i in range(iterations):
        if use_events:
            assert ev_start is not None and ev_end is not None
            ev_start.record()
        else:
            _sync()
            t0 = time.perf_counter()

        with torch.no_grad():
            if graph is not None:
                graph.replay()
                actions = static_actions
            else:
                actions = model.sample_actions(device, observation, num_steps=num_steps)

        if use_events:
            assert ev_start is not None and ev_end is not None
            ev_end.record()
            ev_end.synchronize()
            ms = float(ev_start.elapsed_time(ev_end))
        else:
            _sync()
            ms = (time.perf_counter() - t0) * 1000

        latencies.append(ms)
        print(f"  Iteration {i + 1}: {ms:.1f} ms")

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print("=" * 70)
    print(f"Mean latency:   {np.mean(latencies):.1f} ms")
    print(f"Std:            {np.std(latencies):.1f} ms")
    print(f"Min:            {np.min(latencies):.1f} ms")
    print(f"Max:            {np.max(latencies):.1f} ms")
    print(f"P50:            {np.percentile(latencies, 50):.1f} ms")
    print(f"P95:            {np.percentile(latencies, 95):.1f} ms")
    print(f"Actions shape:  {tuple(actions.shape)}")
    hz = 1000 / np.mean(latencies)
    print(f"Throughput:     {hz:.2f} Hz")
    if batch_size > 1:
        print(f"Samples/s:      {hz * batch_size:.2f} ({batch_size} x {hz:.2f} Hz)")

    # Memory
    print(f"\n{'=' * 70}")
    print("MEMORY")
    print("=" * 70)
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
