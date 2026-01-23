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
sys.path.insert(0, "/sgl-workspace/openpi/src")

import time
import numpy as np
import torch

# Enable aiter attention
os.environ["USE_AITER_ATTENTION"] = "1"
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
    
    # Create model without torch.compile
    print("\nCreating Pi0 model (patching torch.compile)...")
    
    # Temporarily disable torch.compile
    original_compile = torch.compile
    torch.compile = lambda fn, **kwargs: fn
    
    import openpi.models.pi0_config as pi0_config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models import model as _model
    
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=10,
        max_token_len=48,
        dtype='bfloat16',
        paligemma_variant='gemma_2b',
        action_expert_variant='gemma_300m',
        pi05=False,
    )
    
    model = PI0Pytorch(config)
    model = model.to(device)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval()
    
    # Restore torch.compile
    torch.compile = original_compile
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model parameters: {param_count:.2f}B")
    
    # Create observation
    print("\nCreating observation (batch_size=1)...")
    batch_size = 1
    
    images = {
        'base_0_rgb': torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
        'left_wrist_0_rgb': torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
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
    
    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    
    # Benchmark
    num_steps = 10
    warmup = 3
    iterations = 10
    
    print(f"\nDenoising steps: {num_steps}")
    print(f"Warmup: {warmup}, Iterations: {iterations}")
    print("-" * 50)
    
    # Warmup
    print("Warmup...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=num_steps)
    torch.cuda.synchronize()
    
    # Benchmark
    print("Benchmarking...")
    latencies = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
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
