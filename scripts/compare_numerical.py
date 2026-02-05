#!/usr/bin/env python3
"""
Compare numerical outputs between an H200 baseline and the current machine (e.g. MI350).

Typical usage on MI350:
  HIP_VISIBLE_DEVICES=7 python scripts/compare_numerical.py --run --baseline traces/numerical_check_h200_output.pt
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import sys

import torch


def _is_rocm() -> bool:
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def _maybe_patch_transformers_models() -> None:
    """Best-effort: copy our local transformers model replacements into site-packages.

    This matches what `scripts/benchmark_policy_inference.py` does, but keeps this
    script standalone.
    """

    if os.environ.get("AUTO_PATCH_TRANSFORMERS", "1") != "1":
        return

    try:
        import transformers

        repo_root = pathlib.Path(__file__).resolve().parents[1]
        src = (
            repo_root
            / "src"
            / "openpi"
            / "models_pytorch"
            / "transformers_replace"
            / "models"
        )
        if not src.is_dir():
            return

        # transformers.__file__ points at transformers/__init__.py; parent is the package dir.
        dest = pathlib.Path(transformers.__file__).resolve().parent / "models"
        if not dest.is_dir():
            return

        for child in src.iterdir():
            if child.is_dir():
                shutil.copytree(child, dest / child.name, dirs_exist_ok=True)
        print("Patched transformers models from repo", flush=True)
    except Exception as exc:
        print(f"Warning: could not patch transformers models: {exc}", flush=True)


def _safe_load_pt(path: str) -> object:
    # `weights_only=` exists in newer PyTorch; keep backwards compatibility.
    #
    # NOTE: our saved `.pt` dicts may contain `torch.__version__` which is a
    # `TorchVersion` object (subclass of `str`). `weights_only=True` can reject
    # this unless allowlisted, so fall back to `weights_only=False` if needed.
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        try:
            return torch.load(path, weights_only=False)
        except TypeError:
            return torch.load(path)


def _sanitize_tag(s: str) -> str:
    s = s.strip()
    if not s:
        return "cuda0"
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s


def _run_inference(inputs_path: str, output_path: str) -> torch.Tensor:
    # Prefer ROCm-friendly compile mode unless user already set it.
    if "TORCH_COMPILE_MODE" not in os.environ:
        os.environ["TORCH_COMPILE_MODE"] = "default" if _is_rocm() else "reduce-overhead"
    os.environ.setdefault("USE_FUSED_PROJECTIONS", "1")

    _maybe_patch_transformers_models()

    # Import after patching transformers.
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("This numeric check expects a CUDA/ROCm GPU (device=cuda:0).")

    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    inputs = _safe_load_pt(inputs_path)
    assert isinstance(inputs, dict), f"Expected dict inputs, got {type(inputs)}"

    model = PI0Pytorch(Pi0ConfigPytorch())
    model = model.to(device)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval()

    # Optional: fuse projections if supported.
    try:
        model.paligemma_with_expert.fuse_projections(verbose=True)
    except Exception:
        pass

    class SimpleObservation:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    images = {
        "base_0_rgb": inputs["base_0_rgb"].to(device),
        "left_wrist_0_rgb": inputs["left_wrist_0_rgb"].to(device),
        "right_wrist_0_rgb": inputs["right_wrist_0_rgb"].to(device),
    }
    image_masks = {
        "base_0_rgb": inputs["image_mask_base_0_rgb"].to(device),
        "left_wrist_0_rgb": inputs["image_mask_left_wrist_0_rgb"].to(device),
        "right_wrist_0_rgb": inputs["image_mask_right_wrist_0_rgb"].to(device),
    }
    observation = SimpleObservation(
        images=images,
        image_masks=image_masks,
        state=inputs["state"].to(device),
        tokenized_prompt=inputs["tokenized_prompt"].to(device),
        tokenized_prompt_mask=inputs["tokenized_prompt_mask"].to(device),
        token_ar_mask=inputs["token_ar_mask"].to(device),
        token_loss_mask=inputs["token_loss_mask"].to(device),
    )

    print("\nWarmup...", flush=True)
    with torch.inference_mode():
        for _ in range(10):
            _ = model.sample_actions(device, observation, num_steps=10)
        torch.cuda.synchronize()

    print("Running inference...", flush=True)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    with torch.inference_mode():
        actions = model.sample_actions(device, observation, num_steps=10)
        torch.cuda.synchronize()

    actions_cpu = actions.detach().cpu()
    device_name = _sanitize_tag(torch.cuda.get_device_name(0))
    torch.save(
        {
            "actions": actions_cpu,
            "device": device_name,
            "torch_version": torch.__version__,
            "torch_compile_mode": os.environ.get("TORCH_COMPILE_MODE", ""),
        },
        output_path,
    )
    print(f"Output saved to {output_path}", flush=True)
    return actions_cpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=str,
        default="traces/numerical_check_h200_output.pt",
        help="Path to baseline output file (H200)",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default="traces/numerical_check_inputs.pt",
        help="Path to deterministic inputs file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Where to write the current machine output (.pt). Defaults to traces/numerical_check_<device>_output.pt",
    )
    parser.add_argument("--run", action="store_true", help="Run inference on this machine")
    args = parser.parse_args()

    baseline = _safe_load_pt(args.baseline)
    assert isinstance(baseline, dict) and "actions" in baseline, "Baseline file must contain dict['actions']"
    baseline_actions = baseline["actions"]

    out_path = args.output
    if not out_path:
        device_tag = "cuda0"
        if torch.cuda.is_available():
            device_tag = _sanitize_tag(torch.cuda.get_device_name(0))
        out_path = f"traces/numerical_check_{device_tag}_output.pt"

    if args.run:
        pathlib.Path("traces").mkdir(exist_ok=True)
        current_actions = _run_inference(args.inputs, out_path)
    else:
        # Try to load an existing output for this machine.
        cand = pathlib.Path("traces")
        matches = sorted(p for p in cand.glob("numerical_check_*_output.pt") if p.name != pathlib.Path(args.baseline).name)
        if not matches:
            raise SystemExit("No local output found. Run with --run to generate one.")
        current = _safe_load_pt(str(matches[0]))
        assert isinstance(current, dict) and "actions" in current
        current_actions = current["actions"]
        print(f"Loaded {matches[0]}", flush=True)

    diff = (current_actions - baseline_actions).abs()
    rel_diff = diff / (baseline_actions.abs() + 1e-8)

    print("\n" + "=" * 60)
    print("NUMERICAL COMPARISON")
    print("=" * 60)
    print(f"Baseline: {baseline.get('device', 'unknown')} (PyTorch {baseline.get('torch_version', 'unknown')})")
    cur_dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Current:  {cur_dev}")
    print("=" * 60)

    print(f"\nBaseline shape: {tuple(baseline_actions.shape)}")
    print(f"Current shape:  {tuple(current_actions.shape)}")

    print(
        f"\nBaseline stats: min={baseline_actions.min():.6f}, max={baseline_actions.max():.6f}, mean={baseline_actions.mean():.6f}"
    )
    print(f"Current stats:  min={current_actions.min():.6f}, max={current_actions.max():.6f}, mean={current_actions.mean():.6f}")

    print("\nAbsolute difference:")
    print(f"  Max:  {diff.max():.6f}")
    print(f"  Mean: {diff.mean():.6f}")
    print(f"  Std:  {diff.std():.6f}")

    print("\nRelative difference:")
    print(f"  Max:  {rel_diff.max():.6f}")
    print(f"  Mean: {rel_diff.mean():.6f}")

    flat_baseline = baseline_actions.flatten()
    flat_current = current_actions.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(flat_baseline.unsqueeze(0), flat_current.unsqueeze(0))
    print(f"\nCosine similarity: {cos_sim.item():.6f}")

    atol = 1e-3
    rtol = 1e-2
    all_close = torch.allclose(current_actions, baseline_actions, atol=atol, rtol=rtol)
    print(f"\ntorch.allclose(atol={atol}, rtol={rtol}): {all_close}")
    print("=" * 60)


if __name__ == "__main__":
    main()

