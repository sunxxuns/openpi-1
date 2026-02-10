#!/usr/bin/env python3
"""
Capture CUDAGraph replay for rocprofv3 kernel tracing.

Usage:
  rocprofv3 --kernel-trace -d traces/rocprof -o mi300x_pi0_graph -- python scripts/trace_cudagraph.py

This script:
  1. Builds Pi0 model, runs warmup, triggers torch.compile
  2. Captures full sample_actions() into a torch.cuda.CUDAGraph
  3. Warms up graph replay (20 replays)
  4. Runs 5 traced graph replays + 20 timed replays

rocprofv3 --kernel-trace instruments at HIP level and sees every GPU kernel
inside the graph replay (unlike PyTorch profiler which only sees hipGraphLaunch).
"""

import os
import sys
import pathlib
import shutil
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

# ── env defaults ────────────────────────────────────────────────────
os.environ.setdefault("USE_AITER_ATTENTION", "1")
os.environ.setdefault("USE_FUSED_PROJECTIONS", "1")
os.environ.setdefault("USE_AITER_GEMM", "1")
os.environ.setdefault("USE_OPTIMIZED_OPS", "1")
os.environ.setdefault("AITER_MASK_OVERRIDE", "1")
os.environ.setdefault("AITER_EXPERT_MASK_TYPE", "eager")
os.environ.setdefault("OPENPI_INDUCTOR_LOG", "0")
os.environ.setdefault("OPENPI_INDUCTOR_MEMORY_PLANNING", "0")
os.environ.setdefault("OPENPI_DISABLE_COMPILE_AITER_ATTN", "0")
os.environ.setdefault("OPENPI_AITER_ATTN_DIRECT_MHA", "1")
os.environ.setdefault("AUTO_PATCH_TRANSFORMERS", "1")
os.environ.setdefault("OPENPI_MANUAL_CUDAGRAPH", "0")
os.environ.setdefault("AITER_PRESHUFFLE_WEIGHTS", "0")
os.environ.setdefault("OPENPI_SKIP_MASKED_IMAGES", "0")
os.environ.setdefault("OPENPI_NUMERIC_CHECK", "0")
os.environ.setdefault("OPENPI_PROFILE_SHAPES", "0")


def _maybe_extend_aiter_bf16_tuned_gemm_configs():
    if os.environ.get("AITER_CONFIG_GEMM_BF16"):
        return
    repo = pathlib.Path(__file__).resolve().parents[1]
    import torch
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        arch = ""
    if arch.startswith("gfx942"):
        cfg = repo / "configs" / "openpi_bf16_tuned_gemm_mi300.csv"
    else:
        cfg = repo / "configs" / "openpi_bf16_tuned_gemm.csv"
    if not cfg.exists():
        cfg = repo / "configs" / "openpi_bf16_tuned_gemm.csv"
    if not cfg.exists():
        return
    aiter_pkg = None
    try:
        import importlib.util as iu
        spec = iu.find_spec("aiter")
        if spec and spec.submodule_search_locations:
            aiter_pkg = pathlib.Path(list(spec.submodule_search_locations)[0])
    except Exception:
        pass
    if not aiter_pkg:
        os.environ["AITER_CONFIG_GEMM_BF16"] = str(cfg)
        return
    paths = []
    d = aiter_pkg / "configs" / "bf16_tuned_gemm.csv"
    if d.exists():
        paths.append(str(d))
    md = aiter_pkg / "configs" / "model_configs"
    if md.is_dir():
        for p in sorted(md.glob("*bf16_tuned_gemm*.csv")):
            if "untuned" not in p.name:
                paths.append(str(p))
    paths.append(str(cfg))
    os.environ["AITER_CONFIG_GEMM_BF16"] = os.pathsep.join(paths)


def _maybe_patch_transformers():
    if os.environ.get("AUTO_PATCH_TRANSFORMERS", "0") != "1":
        return
    try:
        import transformers
        src = _REPO_ROOT / "src" / "openpi" / "models_pytorch" / "transformers_replace" / "models"
        dest = pathlib.Path(transformers.__file__).resolve().parent / "models"
        for child in src.iterdir():
            if child.is_dir():
                shutil.copytree(child, dest / child.name, dirs_exist_ok=True)
        print("Patched transformers models from repo")
    except Exception as exc:
        print(f"Warning: could not patch transformers: {exc}")


_maybe_extend_aiter_bf16_tuned_gemm_configs()
_maybe_patch_transformers()

import torch
from transformers.models.gemma.modeling_gemma import set_use_aiter_attention
set_use_aiter_attention(True)


def main():
    gpu_id = int(os.environ.get("OPENPI_GPU_ID", "7"))
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    num_steps = 10

    print("=" * 70)
    print("TRACE CUDAGRAPH — Pi0 policy inference (for rocprofv3)")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(gpu_id)} (cuda:{gpu_id})")
    print(f"PyTorch: {torch.__version__}")
    try:
        arch = torch.cuda.get_device_properties(gpu_id).gcnArchName
    except Exception:
        arch = "n/a"
    print(f"Arch: {arch}")
    print(f"Model: Pi0 (pi05=False)")
    print()

    # ── Build model ─────────────────────────────────────────────────
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
    model = PI0Pytorch(config).to(device)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    # aiter GEMM
    if os.environ.get("USE_AITER_GEMM", "0") == "1":
        try:
            from openpi.models_pytorch.aiter_ops import (
                set_use_aiter_gemm, patch_linear_forward, AITER_GEMM_AVAILABLE,
            )
            if AITER_GEMM_AVAILABLE:
                set_use_aiter_gemm(True)
                patch_linear_forward()
                print("Enabled aiter GEMM")
        except Exception as e:
            print(f"Warning: aiter GEMM: {e}")

    # Fuse projections
    try:
        model.paligemma_with_expert.fuse_projections(verbose=True)
    except Exception:
        pass

    # Fuse SigLIP QKV
    if os.environ.get("OPENPI_FUSE_SIGLIP_QKV", "0") == "1":
        try:
            from transformers.models.siglip.modeling_siglip import fuse_siglip_qkv_projections
            fuse_siglip_qkv_projections(model, verbose=True)
        except Exception as e:
            print(f"Warning: SigLIP QKV: {e}")

    # Mask overrides
    if os.environ.get("AITER_MASK_OVERRIDE", "0") == "1":
        try:
            expert_mask = os.environ.get("AITER_EXPERT_MASK_TYPE", "eager").strip().lower()
            for layer in model.paligemma_with_expert.paligemma.language_model.layers:
                layer.self_attn._aiter_mask_type = "full"
            for layer in model.paligemma_with_expert.gemma_expert.model.layers:
                layer.self_attn._aiter_mask_type = expert_mask
            print(f"Applied aiter mask overrides (full/{expert_mask})")
        except Exception:
            pass

    model.eval()
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # ── Build observation (BSZ=1) ───────────────────────────────────
    class Obs:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    obs = Obs(
        images={
            "base_0_rgb": torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device) * 2 - 1,
            "left_wrist_0_rgb": torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device) * 2 - 1,
            "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device),
        },
        image_masks={
            "base_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
            "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=device),
        },
        state=torch.randn(1, 32, dtype=torch.bfloat16, device=device),
        tokenized_prompt=torch.randint(0, 256000, (1, 20), dtype=torch.long, device=device),
        tokenized_prompt_mask=torch.ones(1, 20, dtype=torch.bool, device=device),
        token_ar_mask=torch.ones(1, 20, dtype=torch.int32, device=device),
        token_loss_mask=torch.zeros(1, 20, dtype=torch.bool, device=device),
    )

    # ── Step 1: Warmup (triggers torch.compile) ────────────────────
    print("\n[1/4] Warmup (triggers torch.compile)...")
    for i in range(10):
        with torch.no_grad():
            _ = model.sample_actions(device, obs, num_steps=num_steps)
    torch.cuda.synchronize()
    print("  Warmup done.")

    # ── Step 2: Capture CUDAGraph ───────────────────────────────────
    print("\n[2/4] Capturing CUDAGraph...")
    try:
        from openpi.models_pytorch.rocm_cudagraph_dynamo_patch import (
            patch_dynamo_preserve_global_state_for_rocm_cudagraph_capture,
        )
        patch_dynamo_preserve_global_state_for_rocm_cudagraph_capture()
    except Exception:
        pass

    pool = torch.cuda.graphs.graph_pool_handle()
    graph = torch.cuda.CUDAGraph()

    # Pre-capture warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model.sample_actions(device, obs, num_steps=num_steps)
    torch.cuda.synchronize()

    with torch.cuda.graph(graph, pool=pool):
        static_actions = model.sample_actions(device, obs, num_steps=num_steps)
    torch.cuda.synchronize()
    print("  Graph captured.")

    # ── Step 3: Warm up graph replay ────────────────────────────────
    print("\n[3/4] Warming up graph replay (20 replays)...")
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    print("  Graph replay warm.")

    # ── Step 4: Traced replays + timed replays ──────────────────────
    print("\n[4/4] Running 5 traced replays (rocprofv3 captures these)...")
    torch.cuda.synchronize()
    for i in range(5):
        graph.replay()
        torch.cuda.synchronize()
        print(f"  Traced replay {i+1}/5 done")

    print("\n  Running 20 timed replays...")
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    latencies = []
    for i in range(20):
        ev_s.record()
        graph.replay()
        ev_e.record()
        ev_e.synchronize()
        ms = ev_s.elapsed_time(ev_e)
        latencies.append(ms)
        print(f"  Replay {i+1:2d}: {ms:.1f} ms")

    import numpy as np
    mean_ms = np.mean(latencies)
    print(f"\n  Mean: {mean_ms:.1f} ms  P50: {np.percentile(latencies, 50):.1f} ms  "
          f"P95: {np.percentile(latencies, 95):.1f} ms  -> {1000/mean_ms:.1f} Hz")

    print("\n" + "=" * 70)
    print("Done. Now extract trace from rocprofv3 DB:")
    print("  python scripts/extract_rocprof_trace.py traces/rocprof/mi300x_pi0_graph_results.db")
    print("=" * 70)


if __name__ == "__main__":
    main()
