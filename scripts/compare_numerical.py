#!/usr/bin/env python3
"""
Compare numerical outputs between H200 and MI350.

Usage:
    # Generate baseline (run on H200 first):
    python scripts/compare_numerical.py --run --output traces/numerical_check_h200_output.pt
    
    # Compare on MI350:
    python scripts/compare_numerical.py --run --baseline traces/numerical_check_h200_output.pt
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline output file for comparison")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output (default: auto-generated from device name)")
    parser.add_argument("--run", action="store_true", help="Run inference on this machine")
    args = parser.parse_args()
    
    if args.run:
        # Enable deterministic algorithms for reproducibility
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        # Run inference using saved inputs
        os.environ["TORCH_COMPILE_MODE"] = "reduce-overhead"
        os.environ["USE_FUSED_PROJECTIONS"] = "1"
        
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from dataclasses import dataclass
        
        @dataclass
        class Pi0ConfigPytorch:
            action_dim: int = 32
            action_horizon: int = 10
            max_token_len: int = 48
            dtype: str = 'bfloat16'
            paligemma_variant: str = 'gemma_2b'
            action_expert_variant: str = 'gemma_300m'
            pi05: bool = False
        
        device = torch.device("cuda:0")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")
        
        # Load inputs (includes pre-generated noise)
        inputs = torch.load("traces/numerical_check_inputs.pt", weights_only=True)
        
        config = Pi0ConfigPytorch()
        model = PI0Pytorch(config)
        model = model.to(device)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        model.eval()
        
        try:
            model.paligemma_with_expert.fuse_projections(verbose=True)
        except:
            pass
        
        class SimpleObservation:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        images = {
            'base_0_rgb': inputs['base_0_rgb'].to(device),
            'left_wrist_0_rgb': inputs['left_wrist_0_rgb'].to(device),
            'right_wrist_0_rgb': inputs['right_wrist_0_rgb'].to(device),
        }
        image_masks = {
            'base_0_rgb': inputs['image_mask_base_0_rgb'].to(device),
            'left_wrist_0_rgb': inputs['image_mask_left_wrist_0_rgb'].to(device),
            'right_wrist_0_rgb': inputs['image_mask_right_wrist_0_rgb'].to(device),
        }
        
        observation = SimpleObservation(
            images=images,
            image_masks=image_masks,
            state=inputs['state'].to(device),
            tokenized_prompt=inputs['tokenized_prompt'].to(device),
            tokenized_prompt_mask=inputs['tokenized_prompt_mask'].to(device),
            token_ar_mask=inputs['token_ar_mask'].to(device),
            token_loss_mask=inputs['token_loss_mask'].to(device),
        )
        
        # Get pre-stored noise from inputs
        if 'noise' in inputs:
            noise = inputs['noise'].to(device)
            print(f"Using pre-stored noise from inputs file")
        else:
            # Generate deterministic noise if not in inputs
            torch.manual_seed(123)
            noise = torch.randn(1, 10, 32, dtype=torch.float32, device=device)
            print(f"Generated new noise with seed=123")
        
        # Warmup (without deterministic for speed)
        torch.use_deterministic_algorithms(False)
        print("\nWarmup...")
        with torch.inference_mode():
            for _ in range(5):
                _ = model.sample_actions(device, observation, num_steps=10)
        torch.cuda.synchronize()
        
        # Re-enable deterministic for actual run
        torch.use_deterministic_algorithms(True)
        
        # Run inference with pre-stored noise
        print("Running inference with deterministic settings...")
        with torch.inference_mode():
            actions = model.sample_actions(device, observation, num_steps=10, noise=noise)
        torch.cuda.synchronize()
        
        # Determine output file
        if args.output:
            output_file = args.output
        else:
            device_name = torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "_")
            output_file = f"traces/numerical_check_{device_name}_output.pt"
        
        # Save output with noise for verification
        output_data = {
            'actions': actions.cpu(),
            'noise': noise.cpu(),
            'device': torch.cuda.get_device_name(0),
            'torch_version': torch.__version__,
        }
        torch.save(output_data, output_file)
        print(f"Output saved to {output_file}")
        
        print(f"\nOutput shape: {actions.shape}")
        print(f"Output stats: min={actions.min().item():.6f}, max={actions.max().item():.6f}, mean={actions.mean().item():.6f}")
        print(f"First 10 values: {actions[0, 0, :10].cpu().numpy()}")
        
        current_actions = actions.cpu()
        current_noise = noise.cpu()
    else:
        print("Use --run to generate output on this machine")
        return
    
    # Compare with baseline if provided
    if args.baseline and os.path.exists(args.baseline):
        baseline = torch.load(args.baseline, weights_only=True)
        baseline_actions = baseline['actions']
        baseline_noise = baseline.get('noise', None)
        
        print(f"\n{'='*60}")
        print("NUMERICAL COMPARISON")
        print(f"{'='*60}")
        print(f"Baseline: {baseline.get('device', 'unknown')} (PyTorch {baseline.get('torch_version', 'unknown')})")
        print(f"Current:  {torch.cuda.get_device_name(0)}")
        print(f"{'='*60}")
        
        # Verify noise matches
        if baseline_noise is not None:
            noise_match = torch.allclose(current_noise, baseline_noise)
            print(f"\nNoise tensors match: {noise_match}")
            if not noise_match:
                print("WARNING: Noise tensors don't match - results not comparable!")
        
        # Compare outputs
        diff = (current_actions - baseline_actions).abs()
        
        print(f"\nBaseline shape: {baseline_actions.shape}")
        print(f"Current shape:  {current_actions.shape}")
        
        print(f"\nBaseline stats: min={baseline_actions.min():.6f}, max={baseline_actions.max():.6f}, mean={baseline_actions.mean():.6f}")
        print(f"Current stats:  min={current_actions.min():.6f}, max={current_actions.max():.6f}, mean={current_actions.mean():.6f}")
        
        print(f"\nAbsolute difference:")
        print(f"  Max:  {diff.max():.6f}")
        print(f"  Mean: {diff.mean():.6f}")
        print(f"  Std:  {diff.std():.6f}")
        
        # Relative difference
        rel_diff = diff / (baseline_actions.abs() + 1e-8)
        print(f"\nRelative difference:")
        print(f"  Max:  {rel_diff.max():.6f}")
        print(f"  Mean: {rel_diff.mean():.6f}")
        
        # Cosine similarity
        flat_baseline = baseline_actions.flatten()
        flat_current = current_actions.flatten()
        cos_sim = torch.nn.functional.cosine_similarity(flat_baseline.unsqueeze(0), flat_current.unsqueeze(0))
        print(f"\nCosine similarity: {cos_sim.item():.6f}")
        
        # Check if within tolerance
        atol = 1e-3
        rtol = 1e-2
        all_close = torch.allclose(current_actions, baseline_actions, atol=atol, rtol=rtol)
        print(f"\ntorch.allclose(atol={atol}, rtol={rtol}): {all_close}")
        
        if cos_sim > 0.99:
            print("\n✓ PASSED: Outputs are numerically similar (cosine > 0.99)")
        else:
            print("\n✗ FAILED: Outputs differ significantly")

if __name__ == "__main__":
    main()
