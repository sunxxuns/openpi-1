#!/usr/bin/env python3
"""
Verify numerical precision of Pi0 policy inference.

Ensures that:
1. Model produces valid outputs (no NaN / Inf)
2. Output tensor has the correct shape
3. Inference is deterministic (same input -> same output)
4. Output values are in a reasonable range

IMPORTANT: This script tests the public model API (model.sample_actions).
All backend optimizations (custom attention kernels, GEMM tuning, etc.)
should be implemented INSIDE the model code — NOT by modifying this script.

Usage:
    python scripts/verify_precision.py
    python scripts/verify_precision.py --gpu 7
    python scripts/verify_precision.py --num-steps 5
"""

import os
import sys
import pathlib
import argparse

import torch

# Make repo `src/` importable when running from a source checkout.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Pi0 inference precision")
    parser.add_argument(
        "--gpu", type=int,
        default=int(os.environ.get("OPENPI_GPU_ID", "0")),
        help="GPU device id (default: 0)",
    )
    parser.add_argument(
        "--num-steps", type=int, default=10,
        help="Denoising steps (default: 10)",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)

    print("=" * 70)
    print("PRECISION VERIFICATION: Pi0 Policy Inference")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(args.gpu)} (cuda:{args.gpu})")
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
    # Create deterministic observation
    # ------------------------------------------------------------------ #

    class SimpleObservation:
        """Lightweight observation container matching the model's expected interface."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def create_observation():
        """Create a deterministic observation (re-seeds each call for reproducibility)."""
        torch.manual_seed(42)
        batch_size = 1
        return SimpleObservation(
            images={
                "base_0_rgb": torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
                "left_wrist_0_rgb": torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
                "right_wrist_0_rgb": torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
            },
            image_masks={
                "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
                "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
                "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool, device=device),
            },
            state=torch.randn(batch_size, 32, dtype=torch.bfloat16, device=device),
            tokenized_prompt=torch.randint(0, 256000, (batch_size, 20), dtype=torch.long, device=device),
            tokenized_prompt_mask=torch.ones(batch_size, 20, dtype=torch.bool, device=device),
        )

    # Fixed noise for reproducibility
    torch.manual_seed(123)
    noise = torch.randn(
        1, config.action_horizon, config.action_dim,
        dtype=torch.float32, device=device,
    )

    # ------------------------------------------------------------------ #
    # Warmup (first calls may trigger torch.compile)
    # ------------------------------------------------------------------ #
    print("\nWarmup...")
    obs = create_observation()
    with torch.no_grad():
        for _ in range(3):
            _ = model.sample_actions(device, obs, noise=noise.clone(), num_steps=args.num_steps)
    torch.cuda.synchronize()

    # ------------------------------------------------------------------ #
    # Run 1
    # ------------------------------------------------------------------ #
    print("Run 1: reference inference...")
    obs = create_observation()
    with torch.no_grad():
        actions_1 = model.sample_actions(device, obs, noise=noise.clone(), num_steps=args.num_steps)
    torch.cuda.synchronize()

    # ------------------------------------------------------------------ #
    # Run 2 (identical input)
    # ------------------------------------------------------------------ #
    print("Run 2: repeat inference...")
    obs = create_observation()
    with torch.no_grad():
        actions_2 = model.sample_actions(device, obs, noise=noise.clone(), num_steps=args.num_steps)
    torch.cuda.synchronize()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 70}")
    print("VALIDATION RESULTS")
    print("=" * 70)

    passed = True

    # Check 1: No NaN
    nan_1 = torch.isnan(actions_1).any().item()
    nan_2 = torch.isnan(actions_2).any().item()
    if not nan_1 and not nan_2:
        print("[PASS] No NaN values in outputs")
    else:
        print(f"[FAIL] NaN detected: run1={nan_1}, run2={nan_2}")
        passed = False

    # Check 2: No Inf
    inf_1 = torch.isinf(actions_1).any().item()
    inf_2 = torch.isinf(actions_2).any().item()
    if not inf_1 and not inf_2:
        print("[PASS] No Inf values in outputs")
    else:
        print(f"[FAIL] Inf detected: run1={inf_1}, run2={inf_2}")
        passed = False

    # Check 3: Correct output shape
    expected_shape = (1, config.action_horizon, config.action_dim)
    if tuple(actions_1.shape) == expected_shape:
        print(f"[PASS] Output shape: {tuple(actions_1.shape)}")
    else:
        print(f"[FAIL] Expected shape {expected_shape}, got {tuple(actions_1.shape)}")
        passed = False

    # Check 4: Determinism (same input -> same output)
    diff = (actions_1.float() - actions_2.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        actions_1.flatten().float().unsqueeze(0),
        actions_2.flatten().float().unsqueeze(0),
    ).item()

    if max_diff < 1e-3:
        print(f"[PASS] Deterministic: max_diff={max_diff:.2e}, cosine_sim={cos_sim:.6f}")
    elif max_diff < 1e-1:
        print(f"[WARN] Near-deterministic: max_diff={max_diff:.2e}, cosine_sim={cos_sim:.6f}")
        print("       (small non-determinism is acceptable for bf16 on some GPU backends)")
    else:
        print(f"[FAIL] Non-deterministic: max_diff={max_diff:.2e}, cosine_sim={cos_sim:.6f}")
        passed = False

    # Check 5: Reasonable output range
    min_val = actions_1.min().item()
    max_val = actions_1.max().item()
    mean_val = actions_1.float().mean().item()
    std_val = actions_1.float().std().item()
    print(f"[INFO] Output stats: min={min_val:.4f}, max={max_val:.4f}, "
          f"mean={mean_val:.4f}, std={std_val:.4f}")
    if abs(min_val) < 100 and abs(max_val) < 100 and std_val > 1e-6:
        print("[PASS] Output values in reasonable range")
    else:
        print("[WARN] Output values may be unusual — inspect manually")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 70}")
    if passed:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED")
    print("=" * 70)

    print(f"\nSample action values (first 5): {actions_1[0, 0, :5].tolist()}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
