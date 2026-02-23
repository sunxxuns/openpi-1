#!/usr/bin/env python3
"""Benchmark Pi0 inference performance."""

import sys
sys.path.insert(0, "/workspace/openpi-amd/src")

import torch
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


@dataclass
class SimpleConfig:
    """Simplified config without heavy dependencies."""
    action_dim: int = 32
    action_horizon: int = 50
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m_lora"
    dtype: str = "bfloat16"
    pi05: bool = False


@dataclass
class MockObservation:
    """Mock observation for DROID deployment."""
    images: Dict[str, torch.Tensor]
    image_masks: Dict[str, torch.Tensor]
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor
    token_loss_mask: torch.Tensor
    state: torch.Tensor


def create_mock_observation(batch_size=1, device="cuda"):
    """Create a mock observation for DROID deployment."""
    # 3 cameras, 224x224, 3 channels - use expected keys from preprocessing
    # Use channels-first format [B, C, H, W] as expected by vision model
    image_keys = ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
    images = {
        key: torch.randn(batch_size, 3, 224, 224, device=device)
        for key in image_keys
    }
    image_masks = {key: torch.ones(batch_size, dtype=torch.bool, device=device) for key in image_keys}
    
    # Tokenized prompt (arbitrary length, padded to 64)
    prompt_len = 64
    tokenized_prompt = torch.randint(0, 256000, (batch_size, prompt_len), device=device)
    tokenized_prompt_mask = torch.ones(batch_size, prompt_len, dtype=torch.bool, device=device)
    token_ar_mask = torch.zeros(batch_size, prompt_len, dtype=torch.int32, device=device)  # AR mask
    token_loss_mask = torch.ones(batch_size, prompt_len, dtype=torch.bool, device=device)  # Loss mask
    
    # State dimension - should match action_dim (32)
    state = torch.randn(batch_size, 32, device=device)
    
    return MockObservation(
        images=images,
        image_masks=image_masks,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
        state=state,
    )


def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Create config for Pi0
    config = SimpleConfig()
    
    print("Creating model...")
    model = PI0Pytorch(config).to(device)
    model.eval()
    
    # Create mock observation
    obs = create_mock_observation(batch_size=1, device=device)
    
    print("Warming up...")
    # Warmup runs
    with torch.no_grad():
        for _ in range(20):
            _ = model.sample_actions(device, obs, num_steps=10)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    print("Benchmarking sample_actions...")
    # Timing with CUDA events
    num_iterations = 100
    times = []
    
    for _ in range(num_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            _ = model.sample_actions(device, obs, num_steps=10)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        times.append(elapsed_ms)
    
    times = np.array(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)
    mean = np.mean(times)
    std = np.std(times)
    
    print(f"\n=== Benchmark Results (num_steps=10, bsz=1, bfloat16) ===")
    print(f"Iterations: {num_iterations}")
    print(f"Mean:   {mean:.2f} ms")
    print(f"Std:    {std:.2f} ms")
    print(f"P50:    {p50:.2f} ms")
    print(f"P95:    {p95:.2f} ms")
    print(f"P99:    {p99:.2f} ms")
    print(f"\nTarget: ≤27.00 ms p50")
    print(f"Status: {'✓ PASS' if p50 <= 27.0 else '✗ FAIL'}")


if __name__ == "__main__":
    benchmark()