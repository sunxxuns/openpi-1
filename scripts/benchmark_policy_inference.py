#!/usr/bin/env python3
"""
Benchmark full Pi0 policy inference on AMD MI350.

Tests the complete inference pipeline:
- Image encoding (SigLIP vision tower)
- Language encoding (Gemma embedding)
- Prefill (KV cache generation for images + text)
- Denoising steps (10 steps of action generation)

This measures realistic end-to-end latency for robot control at batch_size=1.
"""

import os
import sys
import pathlib
import shutil
sys.path.insert(0, "/sgl-workspace/openpi/src")

import time
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

# Enable optimizations by default (can be overridden via env)
os.environ.setdefault("USE_AITER_ATTENTION", "1")
os.environ.setdefault("USE_FUSED_PROJECTIONS", "1")
os.environ.setdefault("USE_AITER_GEMM", "1")
os.environ.setdefault("USE_OPTIMIZED_OPS", "1")
os.environ.setdefault("AITER_MASK_OVERRIDE", "1")
os.environ.setdefault("AITER_EXPERT_MASK_TYPE", "eager")  # eager|full (benchmark-only)
os.environ.setdefault("OPENPI_INDUCTOR_LOG", "0")
os.environ.setdefault("AUTO_PATCH_TRANSFORMERS", "1")
os.environ.setdefault("OPENPI_MANUAL_CUDAGRAPH", "0")  # capture+replay full sample_actions (best effort)
os.environ.setdefault("AITER_PRESHUFFLE_WEIGHTS", "0")  # enable bpreshuffle for eligible Linear weights
os.environ.setdefault("OPENPI_NUMERIC_CHECK", "0")  # compare call vs graph replay numerics
os.environ.setdefault("OPENPI_PROFILE_SHAPES", "0")  # print op tables grouped by input shapes


def _maybe_patch_transformers():
    """Copy local transformers replacements into site-packages if enabled."""
    if os.environ.get("AUTO_PATCH_TRANSFORMERS", "0") != "1":
        return
    try:
        import transformers

        src = pathlib.Path(__file__).resolve().parents[1] / "src" / "openpi" / "models_pytorch" / "transformers_replace" / "models"
        # transformers.__file__ points at transformers/__init__.py
        # We want <site-packages>/transformers/models/, i.e. parent is the transformers package dir.
        dest = pathlib.Path(transformers.__file__).resolve().parent / "models"
        for child in src.iterdir():
            if child.is_dir():
                shutil.copytree(child, dest / child.name, dirs_exist_ok=True)
        print("Patched transformers models from repo")
    except Exception as exc:
        print(f"Warning: could not patch transformers models: {exc}")


_maybe_patch_transformers()
from transformers.models.gemma.modeling_gemma import set_use_aiter_attention
set_use_aiter_attention(True)


def main():
    print("=" * 70)
    print("PI0 FULL POLICY INFERENCE BENCHMARK - AMD MI350")
    print("=" * 70)
    
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Aiter Flash Attention: Enabled")
    
    # Create model with torch.compile (reduce-overhead mode for CUDA graphs)
    print("\nCreating Pi0 model...")
    
    from dataclasses import dataclass
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    
    # Create a minimal config class for PyTorch (avoiding JAX/flax imports)
    @dataclass
    class Pi0ConfigPytorch:
        action_dim: int = 32
        action_horizon: int = 10
        max_token_len: int = 48
        dtype: str = 'bfloat16'
        paligemma_variant: str = 'gemma_2b'
        action_expert_variant: str = 'gemma_300m'
        pi05: bool = False
    
    config = Pi0ConfigPytorch()
    
    model = PI0Pytorch(config)
    model = model.to(device)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    # Enable aiter GEMM for optimized matrix multiply (if available)
    if os.environ.get("USE_AITER_GEMM", "0") == "1":
        try:
            from openpi.models_pytorch.aiter_ops import (
                set_use_aiter_gemm,
                patch_linear_forward,
                preshuffle_linear_weights_for_aiter,
                AITER_GEMM_AVAILABLE,
            )
            if AITER_GEMM_AVAILABLE:
                set_use_aiter_gemm(True)
                patch_linear_forward()
                print("Enabled aiter GEMM for optimized matrix multiply")

                if os.environ.get("AITER_PRESHUFFLE_WEIGHTS", "0") == "1":
                    count = preshuffle_linear_weights_for_aiter(model)
                    print(f"Pre-shuffled {count} Linear weights for aiter (bpreshuffle)")
            else:
                print("Warning: aiter GEMM not available")
        except Exception as e:
            print(f"Warning: Could not enable aiter GEMM: {e}")

    # Fuse linear projections to reduce kernel launch overhead
    try:
        model.paligemma_with_expert.fuse_projections(verbose=True)
    except Exception as e:
        if os.environ.get("USE_FUSED_PROJECTIONS", "0") == "1":
            print(f"Warning: Could not fuse projections: {e}")

    # Skip expensive mask checks in aiter attention (benchmark-only)
    if os.environ.get("AITER_MASK_OVERRIDE", "0") == "1":
        try:
            expert_mask_type = os.environ.get("AITER_EXPERT_MASK_TYPE", "eager")
            for layer in model.paligemma_with_expert.paligemma.language_model.layers:
                layer.self_attn._aiter_mask_type = "full"  # full bidirectional
            for layer in model.paligemma_with_expert.gemma_expert.model.layers:
                layer.self_attn._aiter_mask_type = expert_mask_type
            print(f"Applied aiter mask overrides (full/{expert_mask_type})")
        except Exception as e:
            print(f"Warning: Could not apply aiter mask overrides: {e}")
    model.eval()
    
    # torch.compile is applied in PI0Pytorch.__init__ with mode from TORCH_COMPILE_MODE env var
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    default_mode = "default" if is_rocm else "reduce-overhead"
    compile_mode = os.environ.get("TORCH_COMPILE_MODE", default_mode)
    print(f"torch.compile mode: {compile_mode}")
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model parameters: {param_count:.2f}B")
    
    # Create observation
    print("\nCreating observation (batch_size=1)...")
    batch_size = 1
    
    # Simple observation class for PyTorch benchmark
    class SimpleObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    images = {
        'base_0_rgb': torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device) * 2 - 1,
        'left_wrist_0_rgb': torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device) * 2 - 1,
        'right_wrist_0_rgb': torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
    }
    image_masks = {
        'base_0_rgb': torch.ones(batch_size, dtype=torch.bool, device=device),
        'left_wrist_0_rgb': torch.ones(batch_size, dtype=torch.bool, device=device),
        'right_wrist_0_rgb': torch.zeros(batch_size, dtype=torch.bool, device=device),
    }
    state = torch.randn(batch_size, 32, dtype=torch.bfloat16, device=device)
    tokenized_prompt = torch.randint(0, 256000, (batch_size, 20), dtype=torch.long, device=device)
    tokenized_prompt_mask = torch.ones(batch_size, 20, dtype=torch.bool, device=device)
    token_ar_mask = torch.ones(batch_size, 20, dtype=torch.int32, device=device)
    token_loss_mask = torch.zeros(batch_size, 20, dtype=torch.bool, device=device)
    
    observation = SimpleObservation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
    )
    
    # Benchmark (allow env overrides for faster iterations)
    num_steps = int(os.environ.get("NUM_STEPS", "10"))
    warmup = int(os.environ.get("WARMUP", "10"))
    iterations = int(os.environ.get("ITERATIONS", "30"))
    
    print(f"\nDenoising steps: {num_steps}")
    print(f"Warmup: {warmup}, Iterations: {iterations}")
    print("-" * 50)
    
    # Warmup
    print("Warmup...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=num_steps)
    torch.cuda.synchronize()

    # Optional: numeric check (call vs graph replay) using fixed noise
    numeric_check = os.environ.get("OPENPI_NUMERIC_CHECK", "0") == "1"
    fixed_noise = None
    if numeric_check:
        print("\nNumeric check enabled (OPENPI_NUMERIC_CHECK=1)")
        # Make the reference and graph run identical by fixing the input noise.
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        fixed_noise = torch.randn(
            batch_size, config.action_horizon, config.action_dim, device=device, dtype=torch.float32
        )

    # Optional: manual cudagraph capture+replay to reduce CPU/launch overhead
    # (This is separate from Inductor cudagraphs; it captures the entire callable.)
    use_manual_graph = os.environ.get("OPENPI_MANUAL_CUDAGRAPH", "0") == "1"
    graph = None
    static_actions = None
    static_noise = None
    if use_manual_graph:
        print("\nManual CUDAGraph capture enabled (OPENPI_MANUAL_CUDAGRAPH=1)")
        try:
            # Make ROCm graph capture work with torch.compile by skipping CUDA RNG
            # state preservation inside Dynamo while a stream capture is active.
            try:
                from openpi.models_pytorch.rocm_cudagraph_dynamo_patch import (
                    patch_dynamo_preserve_global_state_for_rocm_cudagraph_capture,
                )

                patch_dynamo_preserve_global_state_for_rocm_cudagraph_capture()
            except Exception:
                pass

            # Use a private pool so allocations during capture are replay-safe.
            pool = torch.cuda.graphs.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()

            # Warm up with the *exact* callable we will capture/replay.
            with torch.no_grad():
                for _ in range(5):
                    _ = model.sample_actions(
                        device, observation, noise=fixed_noise, num_steps=num_steps
                    ) if numeric_check else model.sample_actions(
                        device, observation, num_steps=num_steps
                    )
            torch.cuda.synchronize()

            print("Capturing graph (1 iteration)...")
            with torch.cuda.graph(graph, pool=pool):
                static_actions = model.sample_actions(
                    device, observation, noise=fixed_noise, num_steps=num_steps
                ) if numeric_check else model.sample_actions(
                    device, observation, num_steps=num_steps
                )
            torch.cuda.synchronize()
            print("Graph capture succeeded.")
        except Exception as e:
            import traceback

            print(f"Warning: manual CUDAGraph capture failed, falling back to normal run: {e}")
            traceback.print_exc()
            graph = None
            static_actions = None

    if numeric_check:
        with torch.no_grad():
            ref = model.sample_actions(device, observation, noise=fixed_noise, num_steps=num_steps)
            torch.cuda.synchronize()

            if graph is not None:
                graph.replay()
                got = static_actions
                torch.cuda.synchronize()
            else:
                got = model.sample_actions(device, observation, noise=fixed_noise, num_steps=num_steps)
                torch.cuda.synchronize()

        ref_f = ref.float()
        got_f = got.float()
        diff = (ref_f - got_f).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        denom = ref_f.abs().clamp_min(1e-6)
        max_rel = float((diff / denom).max().item())
        print(f"Numeric check: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} max_rel={max_rel:.3e}")

        # Tight-ish defaults for BF16 end-to-end; can be overridden.
        atol = float(os.environ.get("OPENPI_NUMERIC_ATOL", "5e-2"))
        rtol = float(os.environ.get("OPENPI_NUMERIC_RTOL", "5e-2"))
        torch.testing.assert_close(got_f, ref_f, rtol=rtol, atol=atol)
        print(f"Numeric check PASSED (rtol={rtol:g}, atol={atol:g})")
    
    # Optional profiling (set PROFILE=1)
    if os.environ.get("PROFILE", "0") == "1":
        trace_dir = os.environ.get("PROFILE_DIR", "traces")
        os.makedirs(trace_dir, exist_ok=True)
        profile_replay = os.environ.get("PROFILE_GRAPH_REPLAY", "0") == "1"
        profile_shapes = os.environ.get("OPENPI_PROFILE_SHAPES", "0") == "1"
        trace_suffix = "replay" if (profile_replay and graph is not None) else "call"
        trace_path = os.path.join(
            trace_dir, f"policy_inference_compiled_{compile_mode}_{trace_suffix}.json"
        )
        print("\nProfiling (1 iteration)...")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                if profile_replay and graph is not None:
                    graph.replay()
                    _ = static_actions
                else:
                    _ = model.sample_actions(device, observation, num_steps=num_steps)
            torch.cuda.synchronize()
        prof.export_chrome_trace(trace_path)
        trace_size_mb = os.path.getsize(trace_path) / 1024 / 1024
        print(f"Trace saved: {trace_path} ({trace_size_mb:.1f} MB)")
        print("\nTop ops by self CUDA time:")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
        if profile_shapes:
            print("\nTop ops by self CUDA time (grouped by input shape):")
            try:
                print(
                    prof.key_averages(group_by_input_shape=True).table(
                        sort_by="self_cuda_time_total", row_limit=40
                    )
                )
            except Exception as e:
                print(f"(could not group by input shape: {e})")

    # Benchmark
    print("Benchmarking...")
    latencies = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            if graph is not None:
                graph.replay()
                actions = static_actions
            else:
                actions = model.sample_actions(device, observation, num_steps=num_steps)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        print(f"  Iteration {i+1}: {latencies[-1]:.1f} ms")
    
    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print("=" * 70)
    print(f"Mean latency:   {np.mean(latencies):.1f} ms")
    print(f"Std:            {np.std(latencies):.1f} ms")
    print(f"Min:            {np.min(latencies):.1f} ms")
    print(f"Max:            {np.max(latencies):.1f} ms")
    print(f"P50:            {np.percentile(latencies, 50):.1f} ms")
    print(f"P95:            {np.percentile(latencies, 95):.1f} ms")
    print(f"Actions shape:  {tuple(actions.shape)}")
    print(f"Throughput:     {1000/np.mean(latencies):.2f} Hz")
    
    # Memory
    print(f"\n{'='*70}")
    print("MEMORY")
    print("=" * 70)
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
