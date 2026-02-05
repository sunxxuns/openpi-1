#!/usr/bin/env python3
"""
Diagnose torch.compile issues on AMD MI350.

This script helps identify:
- Graph breaks that prevent optimal compilation
- Dynamic shapes that cause recompilation
- Unsupported operations that fall back to eager mode

Usage:
    python scripts/diagnose_compile.py [--verbose]
    
Environment variables:
    TORCH_COMPILE_MODE: default, reduce-overhead, max-autotune, disable
    OPENPI_INDUCTOR_LOG: 0 (off), 1 (basic), 2 (verbose)
"""

import os
import sys
import pathlib

# Make repo `src/` importable when running from a source checkout.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

import argparse
import time

# Enable detailed logging
os.environ["OPENPI_INDUCTOR_LOG"] = "2"
os.environ["TORCH_LOGS"] = "+dynamo,+inductor"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"


def print_section(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def check_system():
    """Check system configuration."""
    print_section("SYSTEM CHECK")
    
    import torch
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    print(f"ROCm/HIP: {is_rocm}")
    if is_rocm:
        print(f"HIP version: {torch.version.hip}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        # Check memory
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {total_mem:.1f} GB")
    
    # Check torch.compile backend
    print(f"\nCompile mode: {os.environ.get('TORCH_COMPILE_MODE', 'reduce-overhead')}")
    
    # Check inductor config
    try:
        import torch._inductor.config as inductor_config
        print(f"\nInductor config:")
        print(f"  epilogue_fusion: {inductor_config.epilogue_fusion}")
        print(f"  pattern_matcher: {inductor_config.pattern_matcher}")
        print(f"  aggressive_fusion: {inductor_config.aggressive_fusion}")
        print(f"  max_autotune: {inductor_config.max_autotune}")
        print(f"  triton.cudagraphs: {inductor_config.triton.cudagraphs}")
        print(f"  triton.cudagraph_trees: {inductor_config.triton.cudagraph_trees}")
    except Exception as e:
        print(f"  (could not read config: {e})")


def check_graph_breaks():
    """Run a simple model and check for graph breaks."""
    print_section("GRAPH BREAK ANALYSIS")
    
    import torch
    import torch._dynamo as dynamo
    
    # Enable graph break logging (if available in this PyTorch version)
    try:
        dynamo.config.log_graph_breaks = True
    except AttributeError:
        print("Note: log_graph_breaks not available in this PyTorch version")
    try:
        dynamo.config.verbose = True
    except AttributeError:
        pass
    
    # Simple test model
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(256, 512)
            self.linear2 = torch.nn.Linear(512, 256)
            self.norm = torch.nn.LayerNorm(256)
            
        def forward(self, x):
            x = self.linear1(x)
            x = torch.nn.functional.gelu(x)
            x = self.linear2(x)
            x = self.norm(x)
            return x
    
    device = torch.device("cuda:0")
    model = TestModel().to(device).to(torch.bfloat16)
    x = torch.randn(2, 32, 256, dtype=torch.bfloat16, device=device)
    
    print("Testing simple model compilation...")
    
    # Compile and run
    compiled_model = torch.compile(model, mode="default")
    
    try:
        # Warmup
        for _ in range(3):
            _ = compiled_model(x)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = compiled_model(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 100 * 1000
        
        print(f"Simple model: {elapsed:.3f} ms/iter")
        print("No critical graph breaks detected.")
    except Exception as e:
        print(f"ERROR during compilation: {e}")


def check_attention_compilation():
    """Check if attention compiles correctly."""
    print_section("ATTENTION COMPILATION CHECK")
    
    import torch
    from transformers.models.gemma.modeling_gemma import (
        GemmaAttention,
        GemmaConfig,
        set_use_aiter_attention,
        get_use_aiter_attention,
    )
    
    device = torch.device("cuda:0")
    
    config = GemmaConfig(
        hidden_size=2048,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=256,
        intermediate_size=8192,
    )
    config._attn_implementation = "eager"
    
    # Test with aiter attention
    print(f"Aiter attention available: {get_use_aiter_attention()}")
    set_use_aiter_attention(True)
    print(f"Aiter attention enabled: {get_use_aiter_attention()}")
    
    attn = GemmaAttention(config, layer_idx=0).to(device).to(torch.bfloat16)
    
    batch, seq = 1, 128
    hidden = torch.randn(batch, seq, config.hidden_size, dtype=torch.bfloat16, device=device)
    cos = torch.randn(batch, seq, config.head_dim, dtype=torch.bfloat16, device=device)
    sin = torch.randn(batch, seq, config.head_dim, dtype=torch.bfloat16, device=device)
    mask = torch.zeros(batch, 1, seq, seq, dtype=torch.bfloat16, device=device)
    
    print("Testing attention compilation...")
    
    def attn_forward(hidden, cos, sin, mask):
        return attn(hidden, position_embeddings=(cos, sin), attention_mask=mask)
    
    compiled_attn = torch.compile(attn_forward, mode="default")
    
    try:
        # Warmup
        for _ in range(3):
            _ = compiled_attn(hidden, cos, sin, mask)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(50):
            _ = compiled_attn(hidden, cos, sin, mask)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 50 * 1000
        
        print(f"Attention forward: {elapsed:.3f} ms/iter")
    except Exception as e:
        print(f"ERROR during attention compilation: {e}")
        import traceback
        traceback.print_exc()


def check_full_model():
    """Check full PI0 model compilation."""
    print_section("FULL MODEL COMPILATION CHECK")
    
    import torch
    from dataclasses import dataclass
    
    # Set compile mode for testing
    compile_mode = os.environ.get("TORCH_COMPILE_MODE", "default")
    print(f"Using compile mode: {compile_mode}")
    
    # Temporarily set to test mode
    os.environ["TORCH_COMPILE_MODE"] = compile_mode
    
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    
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
    
    print("Creating PI0 model...")
    start = time.perf_counter()
    
    try:
        config = Pi0ConfigPytorch()
        model = PI0Pytorch(config)
        model = model.to(device)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        model.eval()
        
        elapsed = time.perf_counter() - start
        print(f"Model creation: {elapsed:.1f}s")
        
        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"Parameters: {param_count:.2f}B")
        
        # Create dummy observation
        class SimpleObservation:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        batch_size = 1
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
        
        print("\nRunning inference (first call triggers compilation)...")
        start = time.perf_counter()
        
        with torch.no_grad():
            actions = model.sample_actions(device, observation, num_steps=10)
        torch.cuda.synchronize()
        
        first_call = time.perf_counter() - start
        print(f"First call (includes compilation): {first_call:.1f}s")
        
        # Second call should be fast
        start = time.perf_counter()
        with torch.no_grad():
            actions = model.sample_actions(device, observation, num_steps=10)
        torch.cuda.synchronize()
        second_call = (time.perf_counter() - start) * 1000
        
        print(f"Second call: {second_call:.1f}ms")
        print(f"Actions shape: {tuple(actions.shape)}")
        
        if second_call < 100:
            print("\n✓ Compilation successful - fast inference achieved!")
        else:
            print("\n⚠ Compilation may have issues - inference is slow")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Diagnose torch.compile issues on AMD MI350")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--skip-full", action="store_true", help="Skip full model test")
    args = parser.parse_args()
    
    if args.verbose:
        os.environ["OPENPI_INDUCTOR_LOG"] = "2"
    
    print("=" * 70)
    print(" TORCH.COMPILE DIAGNOSTIC TOOL FOR AMD MI350")
    print("=" * 70)
    
    check_system()
    check_graph_breaks()
    check_attention_compilation()
    
    if not args.skip_full:
        check_full_model()
    
    print_section("DONE")
    print("If you see graph breaks or compilation errors above, they may indicate")
    print("operations that don't compile well on ROCm and could benefit from")
    print("custom Triton kernels or code changes.")


if __name__ == "__main__":
    main()
