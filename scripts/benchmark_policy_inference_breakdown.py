#!/usr/bin/env python3
"""
Latency breakdown of Pi0 policy inference (BSZ=1) on AMD MI300X.

Reports time for each stage:
  1) ViT    – SigLIP vision tower (image encoding)
  2) LLM    – Gemma prefill (KV-cache build from prefix tokens)
  3) Diffusion – 10 denoising steps (action expert)

Two modes:
  A) torch.compile (per-stage timing via CUDA events)
  B) Full CUDAGraph replay (total latency only – fastest)
"""

import os
import sys
import pathlib
import shutil

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

import argparse
import time
import math
import numpy as np
import torch

# ── env defaults (same as main benchmark) ──────────────────────────
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
os.environ.setdefault("OPENPI_NUMERIC_CHECK", "0")
os.environ.setdefault("OPENPI_PROFILE_SHAPES", "0")
os.environ.setdefault("OPENPI_SKIP_MASKED_IMAGES", "0")  # apples-to-apples


def _maybe_extend_aiter_bf16_tuned_gemm_configs():
    if os.environ.get("AITER_CONFIG_GEMM_BF16"):
        return
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    try:
        props = torch.cuda.get_device_properties(0)
        arch = props.gcnArchName.split(":")[0] if hasattr(props, "gcnArchName") else ""
    except Exception:
        arch = ""
    if arch.startswith("gfx942"):
        local_cfg = repo_root / "configs" / "openpi_bf16_tuned_gemm_mi300.csv"
        print(f"[openpi] Detected MI300 ({arch}), using MI300 GEMM config")
    else:
        local_cfg = repo_root / "configs" / "openpi_bf16_tuned_gemm.csv"
        print(f"[openpi] GPU arch={arch}, using default GEMM config")
    if not local_cfg.exists():
        local_cfg = repo_root / "configs" / "openpi_bf16_tuned_gemm.csv"
    if not local_cfg.exists():
        return
    aiter_pkg = None
    try:
        import importlib.util as _iu
        spec = _iu.find_spec("aiter")
        if spec and spec.submodule_search_locations:
            aiter_pkg = pathlib.Path(list(spec.submodule_search_locations)[0])
    except Exception:
        pass
    if aiter_pkg is None:
        os.environ["AITER_CONFIG_GEMM_BF16"] = str(local_cfg)
        return
    default_cfg = aiter_pkg / "configs" / "bf16_tuned_gemm.csv"
    model_cfg_dir = aiter_pkg / "configs" / "model_configs"
    paths = []
    if default_cfg.exists():
        paths.append(str(default_cfg))
    if model_cfg_dir.is_dir():
        for p in sorted(model_cfg_dir.glob("*bf16_tuned_gemm*.csv")):
            if "untuned" in p.name:
                continue
            paths.append(str(p))
    paths.append(str(local_cfg))
    os.environ["AITER_CONFIG_GEMM_BF16"] = os.pathsep.join(paths)


def _maybe_patch_transformers():
    if os.environ.get("AUTO_PATCH_TRANSFORMERS", "0") != "1":
        return
    try:
        import transformers
        src = pathlib.Path(__file__).resolve().parents[1] / "src" / "openpi" / "models_pytorch" / "transformers_replace" / "models"
        dest = pathlib.Path(transformers.__file__).resolve().parent / "models"
        for child in src.iterdir():
            if child.is_dir():
                shutil.copytree(child, dest / child.name, dirs_exist_ok=True)
        print("Patched transformers models from repo")
    except Exception as exc:
        print(f"Warning: could not patch transformers models: {exc}")


_maybe_extend_aiter_bf16_tuned_gemm_configs()
_maybe_patch_transformers()
from transformers.models.gemma.modeling_gemma import set_use_aiter_attention
set_use_aiter_attention(True)


def parse_args():
    p = argparse.ArgumentParser(description="Pi0 policy inference – latency breakdown")
    p.add_argument("--gpu", type=int, default=int(os.environ.get("OPENPI_GPU_ID", "7")))
    p.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "10")))
    p.add_argument("--iterations", type=int, default=int(os.environ.get("ITERATIONS", "30")))
    p.add_argument("--num-steps", type=int, default=int(os.environ.get("NUM_STEPS", "10")))
    return p.parse_args()


def _sync():
    try:
        torch.cuda.current_stream().synchronize()
    except Exception:
        torch.cuda.synchronize()


# ── Timing helpers ──────────────────────────────────────────────────
def _event_pair():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def main():
    args = parse_args()
    gpu_id = args.gpu
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    num_steps = args.num_steps

    print("=" * 70)
    print("PI0 POLICY INFERENCE – LATENCY BREAKDOWN (BSZ=1)")
    print("=" * 70)
    print(f"Device : {torch.cuda.get_device_name(gpu_id)} (cuda:{gpu_id})")
    print(f"PyTorch: {torch.__version__}")
    try:
        arch = torch.cuda.get_device_properties(gpu_id).gcnArchName
    except Exception:
        arch = "n/a"
    print(f"Arch   : {arch}")
    print(f"Model  : Pi0 (pi05=False)")
    print(f"Denoise: {num_steps} steps")
    print()

    # ── Build model (same as main benchmark) ────────────────────────
    from dataclasses import dataclass
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

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
                set_use_aiter_gemm, patch_linear_forward,
                preshuffle_linear_weights_for_aiter, AITER_GEMM_AVAILABLE,
            )
            if AITER_GEMM_AVAILABLE:
                set_use_aiter_gemm(True)
                patch_linear_forward()
                print("Enabled aiter GEMM")
                if os.environ.get("AITER_PRESHUFFLE_WEIGHTS", "0") == "1":
                    count = preshuffle_linear_weights_for_aiter(model)
                    print(f"Pre-shuffled {count} weights")
        except Exception as e:
            print(f"Warning: aiter GEMM: {e}")

    # Fuse projections
    try:
        model.paligemma_with_expert.fuse_projections(verbose=True)
    except Exception as e:
        if os.environ.get("USE_FUSED_PROJECTIONS", "0") == "1":
            print(f"Warning: fuse projections: {e}")

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
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Params : {param_count:.2f}B")

    # ── Build observation (BSZ=1) ───────────────────────────────────
    batch_size = 1

    class SimpleObservation:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

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
    observation = SimpleObservation(
        images=images,
        image_masks=image_masks,
        state=torch.randn(batch_size, 32, dtype=torch.bfloat16, device=device),
        tokenized_prompt=torch.randint(0, 256000, (batch_size, 20), dtype=torch.long, device=device),
        tokenized_prompt_mask=torch.ones(batch_size, 20, dtype=torch.bool, device=device),
        token_ar_mask=torch.ones(batch_size, 20, dtype=torch.int32, device=device),
        token_loss_mask=torch.zeros(batch_size, 20, dtype=torch.bool, device=device),
    )

    # ── Warmup (full sample_actions) ────────────────────────────────
    print("\nWarming up full pipeline ...")
    for _ in range(args.warmup):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=num_steps)
    _sync()
    print("Warmup done.\n")

    # ================================================================
    # PART A – Per-stage breakdown (torch.compile, NO CUDAGraph)
    # ================================================================
    print("=" * 70)
    print("PART A: PER-STAGE LATENCY BREAKDOWN  (torch.compile, no CUDAGraph)")
    print("=" * 70)

    vit_times, llm_times, diff_times, total_times = [], [], [], []

    ev_vit_s, ev_vit_e = _event_pair()
    ev_llm_s, ev_llm_e = _event_pair()
    ev_diff_s, ev_diff_e = _event_pair()
    ev_tot_s, ev_tot_e = _event_pair()

    for i in range(args.iterations):
        with torch.no_grad():
            ev_tot_s.record()

            # ── Stage 1: ViT (SigLIP image encoding + lang embed) ──
            ev_vit_s.record()
            imgs, img_masks_list, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                imgs, img_masks_list, lang_tokens, lang_masks
            )
            ev_vit_e.record()

            # ── Stage 2: LLM prefill (KV-cache build) ──────────────
            ev_llm_s.record()
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
            model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
            _, past_key_values = model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            ev_llm_e.record()

            # ── Stage 3: Diffusion (10 denoising steps) ────────────
            ev_diff_s.record()
            bsize = state.shape[0]
            actions_shape = (bsize, config.action_horizon, config.action_dim)
            noise = model.sample_noise(actions_shape, device)
            dt = -1.0 / num_steps
            times_key = (str(device), int(num_steps))
            t_cache = getattr(model, "_openpi_sample_actions_times", None)
            if not isinstance(t_cache, dict) or t_cache.get("key") != times_key:
                t = torch.arange(num_steps, device=device, dtype=torch.float32) * dt + 1.0
                t_cache = {"key": times_key, "t": t}
                setattr(model, "_openpi_sample_actions_times", t_cache)
            t_schedule = t_cache["t"]

            x_t = noise
            for step in range(num_steps):
                expanded_time = t_schedule[step].expand(bsize)
                v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
                x_t = x_t + (dt * v_t)
            ev_diff_e.record()

            ev_tot_e.record()

        ev_tot_e.synchronize()
        vit_ms = ev_vit_s.elapsed_time(ev_vit_e)
        llm_ms = ev_llm_s.elapsed_time(ev_llm_e)
        diff_ms = ev_diff_s.elapsed_time(ev_diff_e)
        tot_ms = ev_tot_s.elapsed_time(ev_tot_e)

        vit_times.append(vit_ms)
        llm_times.append(llm_ms)
        diff_times.append(diff_ms)
        total_times.append(tot_ms)

        print(f"  iter {i+1:2d}:  ViT {vit_ms:6.1f}   LLM {llm_ms:6.1f}   "
              f"Diff {diff_ms:6.1f}   Total {tot_ms:6.1f} ms")

    print()
    vit_mean = np.mean(vit_times)
    llm_mean = np.mean(llm_times)
    diff_mean = np.mean(diff_times)
    tot_mean = np.mean(total_times)

    print(f"{'Stage':<12} {'Mean (ms)':>10} {'Std':>8} {'% of total':>12}")
    print("-" * 46)
    print(f"{'ViT':<12} {vit_mean:10.1f} {np.std(vit_times):8.1f} {vit_mean/tot_mean*100:11.1f}%")
    print(f"{'LLM prefill':<12} {llm_mean:10.1f} {np.std(llm_times):8.1f} {llm_mean/tot_mean*100:11.1f}%")
    print(f"{'Diffusion':<12} {diff_mean:10.1f} {np.std(diff_times):8.1f} {diff_mean/tot_mean*100:11.1f}%")
    print("-" * 46)
    print(f"{'Total':<12} {tot_mean:10.1f} {np.std(total_times):8.1f} {'100.0':>11}%")
    print(f"\nPer-denoise-step: {diff_mean / num_steps:.1f} ms  ({num_steps} steps)")

    # ================================================================
    # PART B – Full CUDAGraph replay (fastest, headline number)
    # ================================================================
    print()
    print("=" * 70)
    print("PART B: FULL CUDAGRAPH REPLAY  (fastest, headline BSZ=1 latency)")
    print("=" * 70)

    # Re-warmup for graph capture
    for _ in range(5):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=num_steps)
    _sync()

    try:
        from openpi.models_pytorch.rocm_cudagraph_dynamo_patch import (
            patch_dynamo_preserve_global_state_for_rocm_cudagraph_capture,
        )
        patch_dynamo_preserve_global_state_for_rocm_cudagraph_capture()
    except Exception:
        pass

    pool = torch.cuda.graphs.graph_pool_handle()
    graph = torch.cuda.CUDAGraph()

    # Pre-capture warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model.sample_actions(device, observation, num_steps=num_steps)
    _sync()

    print("Capturing CUDAGraph ...")
    with torch.cuda.graph(graph, pool=pool):
        static_actions = model.sample_actions(device, observation, num_steps=num_steps)
    _sync()
    print("Graph capture succeeded.\n")

    ev_s, ev_e = _event_pair()
    graph_latencies = []
    for i in range(args.iterations):
        ev_s.record()
        graph.replay()
        ev_e.record()
        ev_e.synchronize()
        ms = ev_s.elapsed_time(ev_e)
        graph_latencies.append(ms)
        print(f"  iter {i+1:2d}: {ms:.1f} ms")

    graph_mean = np.mean(graph_latencies)
    print(f"\nCUDAGraph BSZ=1 latency: {graph_mean:.1f} ms  "
          f"(P50={np.percentile(graph_latencies,50):.1f}  "
          f"P95={np.percentile(graph_latencies,95):.1f})  "
          f"→ {1000/graph_mean:.1f} Hz")

    # ================================================================
    # Summary
    # ================================================================
    print()
    print("=" * 70)
    print("SUMMARY  –  Pi0 policy inference, BSZ=1, MI300X")
    print("=" * 70)
    print()
    print(f"  Model          : Pi0 (pi05=False, 3.5B params)")
    print(f"  Denoising steps: {num_steps}")
    print(f"  GPU            : {torch.cuda.get_device_name(gpu_id)} ({arch})")
    print()
    print(f"  CUDAGraph total: {graph_mean:.1f} ms  ({1000/graph_mean:.1f} Hz)")
    print()
    print(f"  Breakdown (torch.compile, no graph):")
    print(f"    ViT (SigLIP)   : {vit_mean:5.1f} ms  ({vit_mean/tot_mean*100:.0f}%)")
    print(f"    LLM prefill    : {llm_mean:5.1f} ms  ({llm_mean/tot_mean*100:.0f}%)")
    print(f"    Diffusion x{num_steps:2d}  : {diff_mean:5.1f} ms  ({diff_mean/tot_mean*100:.0f}%)  "
          f"[{diff_mean/num_steps:.1f} ms/step]")
    print(f"    ─────────────────────────────")
    print(f"    Sum            : {tot_mean:5.1f} ms")
    print()
    print(f"  Memory: {torch.cuda.memory_allocated(device)/1e9:.2f} GB allocated, "
          f"{torch.cuda.memory_reserved(device)/1e9:.2f} GB reserved")
    print("=" * 70)


if __name__ == "__main__":
    main()
