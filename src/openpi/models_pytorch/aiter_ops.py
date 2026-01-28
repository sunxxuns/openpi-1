"""Optimized operations for AMD MI350 using Triton kernels."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Global flag to enable optimized ops (Triton-based)
USE_OPTIMIZED_OPS = os.environ.get("USE_OPTIMIZED_OPS", "0") == "1"
TRITON_AVAILABLE = False

try:
    from openpi.models_pytorch.triton_ops import (
        rms_norm_triton,
        gelu_tanh_and_mul_triton,
        silu_and_mul_triton,
        fused_add_rms_norm_triton,
    )
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Legacy aiter support (for flash attention)
AITER_AVAILABLE = False
AITER_GEMM_AVAILABLE = False
try:
    import aiter
    AITER_AVAILABLE = True
    # Check for tuned GEMM
    try:
        from aiter.tuned_gemm import gemm_a16w16
        AITER_GEMM_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass

# Global flag to enable aiter GEMM
USE_AITER_GEMM = os.environ.get("USE_AITER_GEMM", "0") == "1"


def set_use_aiter_ops(enabled: bool):
    """Enable or disable optimized ops globally."""
    global USE_OPTIMIZED_OPS
    USE_OPTIMIZED_OPS = enabled


def set_use_optimized_ops(enabled: bool):
    """Enable or disable optimized ops globally."""
    global USE_OPTIMIZED_OPS
    USE_OPTIMIZED_OPS = enabled


def get_use_aiter_ops() -> bool:
    """Check if optimized ops are enabled."""
    return USE_OPTIMIZED_OPS and TRITON_AVAILABLE


def get_use_optimized_ops() -> bool:
    """Check if optimized ops are enabled."""
    return USE_OPTIMIZED_OPS and TRITON_AVAILABLE


# ============================================================================
# RMSNorm - uses aiter.rms_norm
# ============================================================================

def rms_norm_eager(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard RMSNorm implementation."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight).to(x.dtype)


def rms_norm_optimized(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Triton optimized RMSNorm (3.4x faster)."""
    if TRITON_AVAILABLE:
        return rms_norm_triton(x, weight, eps)
    return rms_norm_eager(x, weight, eps)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm with automatic backend selection."""
    if get_use_optimized_ops():
        return rms_norm_optimized(x, weight, eps)
    return rms_norm_eager(x, weight, eps)


# ============================================================================
# Fused GELU + Mul (for MLP gate)
# ============================================================================

def gelu_tanh_and_mul_eager(x: torch.Tensor) -> torch.Tensor:
    """Standard GELU tanh approximation with mul.
    
    Input x is [batch, seq, 2*hidden] where first half is gate, second is up.
    Output is [batch, seq, hidden].
    """
    hidden_size = x.shape[-1] // 2
    gate = x[..., :hidden_size]
    up = x[..., hidden_size:]
    return F.gelu(gate, approximate='tanh') * up


def gelu_tanh_and_mul_optimized(x: torch.Tensor) -> torch.Tensor:
    """Triton optimized GELU tanh + mul (1.6x faster)."""
    if TRITON_AVAILABLE:
        return gelu_tanh_and_mul_triton(x)
    return gelu_tanh_and_mul_eager(x)


def gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
    """GELU tanh + mul with automatic backend selection."""
    if get_use_optimized_ops():
        return gelu_tanh_and_mul_optimized(x)
    return gelu_tanh_and_mul_eager(x)


# ============================================================================
# Fused SiLU + Mul (alternative activation)
# ============================================================================

def silu_and_mul_eager(x: torch.Tensor) -> torch.Tensor:
    """Standard SiLU with mul.
    
    Input x is [batch, seq, 2*hidden] where first half is gate, second is up.
    """
    hidden_size = x.shape[-1] // 2
    gate = x[..., :hidden_size]
    up = x[..., hidden_size:]
    return F.silu(gate) * up


def silu_and_mul_optimized(x: torch.Tensor) -> torch.Tensor:
    """Triton optimized SiLU + mul (1.4x faster)."""
    if TRITON_AVAILABLE:
        return silu_and_mul_triton(x)
    return silu_and_mul_eager(x)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SiLU + mul with automatic backend selection."""
    if get_use_optimized_ops():
        return silu_and_mul_optimized(x)
    return silu_and_mul_eager(x)


# ============================================================================
# Rotary Position Embedding
# ============================================================================

def apply_rotary_pos_emb_eager(
    q: torch.Tensor,
    k: torch.Tensor, 
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard rotary position embedding."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Rotate half
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_aiter(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aiter optimized rotary position embedding.
    
    Note: aiter.rope_fwd expects different input format, so we need to adapt.
    """
    # aiter rope expects [batch, seq, heads, head_dim] or similar
    # Our input is [batch, heads, seq, head_dim]
    # Need to construct freqs from cos/sin
    
    # For now, fall back to eager if format doesn't match
    # TODO: Implement proper aiter RoPE integration
    return apply_rotary_pos_emb_eager(q, k, cos, sin, unsqueeze_dim)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor, 
    sin: torch.Tensor,
    unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary position embedding with automatic backend selection."""
    if get_use_aiter_ops():
        return apply_rotary_pos_emb_aiter(q, k, cos, sin, unsqueeze_dim)
    return apply_rotary_pos_emb_eager(q, k, cos, sin, unsqueeze_dim)


# ============================================================================
# Fused Add + RMSNorm (for residual connections)
# ============================================================================

def fused_add_rms_norm_eager(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard add + RMSNorm.
    
    Returns (normalized_output, residual_sum).
    """
    hidden = x + residual
    variance = hidden.float().pow(2).mean(-1, keepdim=True)
    normed = hidden * torch.rsqrt(variance + eps)
    return (normed * weight).to(x.dtype), hidden


def fused_add_rms_norm_optimized(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton optimized fused add + RMSNorm (2.8x faster)."""
    if TRITON_AVAILABLE:
        return fused_add_rms_norm_triton(x, residual, weight, eps)
    return fused_add_rms_norm_eager(x, residual, weight, eps)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused add + RMSNorm with automatic backend selection."""
    if get_use_optimized_ops():
        return fused_add_rms_norm_optimized(x, residual, weight, eps)
    return fused_add_rms_norm_eager(x, residual, weight, eps)


# ============================================================================
# Optimized GemmaMLP using fused activations
# ============================================================================

class OptimizedGemmaMLP(nn.Module):
    """GemmaMLP with aiter optimized fused gate activation."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "gelu_pytorch_tanh"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        
        # Fused gate+up projection
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused gate_proj and up_proj
        gate_up = self.gate_up_proj(x)
        
        # Fused activation + multiply
        if self.activation == "gelu_pytorch_tanh":
            hidden = gelu_tanh_and_mul(gate_up)
        elif self.activation == "silu":
            hidden = silu_and_mul(gate_up)
        else:
            # Fallback to separate ops
            gate = gate_up[..., :self.intermediate_size]
            up = gate_up[..., self.intermediate_size:]
            hidden = F.gelu(gate, approximate='tanh') * up
        
        return self.down_proj(hidden)


# ============================================================================
# Fuse MLP weights (gate + up) for GemmaMLP
# ============================================================================

def fuse_mlp_weights(model: nn.Module) -> nn.Module:
    """
    Fuse gate_proj and up_proj weights in GemmaMLP modules.
    
    This creates a combined weight for gate+up projection, enabling:
    1. Single GEMM instead of two
    2. Fused GELU+mul kernel when available
    
    Call this AFTER loading weights but BEFORE torch.compile.
    """
    try:
        from transformers.models.gemma.modeling_gemma import GemmaMLP
    except ImportError:
        print("Warning: Could not import GemmaMLP, skipping fusion")
        return model
    
    fused_count = 0
    
    for _, module in model.named_modules():
        if isinstance(module, GemmaMLP):
            if hasattr(module, "_use_fused") and module._use_fused:
                continue
            fused_weight = torch.cat(
                [module.gate_proj.weight, module.up_proj.weight], dim=0
            )
            module.register_buffer("_fused_gate_up_weight", fused_weight)
            module._use_fused = True
            fused_count += 1
    
    if fused_count > 0:
        print(f"Fused {fused_count} MLP modules")
    
    return model


# ============================================================================
# Aiter GEMM - optimized matrix multiply for AMD
# ============================================================================

def set_use_aiter_gemm(enabled: bool):
    """Enable or disable aiter GEMM globally."""
    global USE_AITER_GEMM
    USE_AITER_GEMM = enabled
    if enabled and not AITER_GEMM_AVAILABLE:
        print("Warning: aiter GEMM not available, falling back to F.linear")


def get_use_aiter_gemm() -> bool:
    """Check if aiter GEMM is enabled."""
    return USE_AITER_GEMM and AITER_GEMM_AVAILABLE


def aiter_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Linear layer using aiter's optimized GEMM kernels.
    
    Prefer aiter's tuned GEMM dispatcher (`aiter.tuned_gemm.gemm_a16w16`) which
    selects among asm/skinny/hipblaslt/torch per-shape based on tuned configs.
    """
    if not AITER_GEMM_AVAILABLE:
        return F.linear(x, weight, bias)

    # Only route bf16/fp16 through tuned GEMM for now.
    # (Other dtypes fall back to PyTorch.)
    if x.dtype not in (torch.bfloat16, torch.float16) or weight.dtype != x.dtype:
        return F.linear(x, weight, bias)

    try:
        # `gemm_a16w16` expects B as [N, K] (nn.Linear.weight layout).
        return gemm_a16w16(x, weight, bias=bias, otype=x.dtype)
    except Exception:
        # Conservative fallback (never fail correctness)
        return F.linear(x, weight, bias)


class AiterLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that uses aiter's tuned GEMM.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if get_use_aiter_gemm():
            return aiter_linear(x, self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)
    
    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "AiterLinear":
        """Convert an existing nn.Linear to AiterLinear."""
        new_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        new_linear.weight = linear.weight
        if linear.bias is not None:
            new_linear.bias = linear.bias
        return new_linear


def replace_linear_with_aiter(model: nn.Module, target_modules: list = None) -> nn.Module:
    """
    Replace nn.Linear modules with AiterLinear for optimized GEMM.
    """
    if not AITER_GEMM_AVAILABLE:
        print("Warning: aiter GEMM not available, skipping replacement")
        return model
    
    replaced_count = 0
    
    def should_replace(name):
        if target_modules is None:
            return True
        return any(t in name for t in target_modules)
    
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_replace(name):
            replacements.append((name, module))
    
    for name, linear in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], AiterLinear.from_linear(linear))
        replaced_count += 1
    
    if replaced_count > 0:
        print(f"Replaced {replaced_count} Linear layers with AiterLinear")
    
    return model


def patch_linear_forward():
    """
    Monkey-patch nn.Linear.forward to use aiter GEMM when enabled.
    """
    if not AITER_GEMM_AVAILABLE:
        print("Warning: aiter GEMM not available, skipping patch")
        return
    
    original_forward = nn.Linear.forward
    
    def patched_forward(self, x):
        if get_use_aiter_gemm():
            w = getattr(self, "_aiter_preshuffled_weight", self.weight)
            return aiter_linear(x, w, self.bias)
        return original_forward(self, x)
    
    nn.Linear.forward = patched_forward
    print("Patched nn.Linear.forward to use aiter GEMM (toggle with set_use_aiter_gemm)")


def preshuffle_linear_weights_for_aiter(
    model: nn.Module,
    *,
    require_multiple: int = 256,
    layout: tuple[int, int] = (16, 16),
) -> int:
    """Pre-shuffle eligible Linear weights for aiter GEMM on gfx950.

    aiter's bf16 asm GEMM path can use pre-shuffled weights (bpreshuffle=True) to
    enable faster kernels on MI350-class GPUs. We keep the original `.weight`
    intact and stash the shuffled version on the module to avoid breaking any
    non-Linear codepaths that might directly read `.weight`.
    """
    if not AITER_GEMM_AVAILABLE:
        return 0

    try:
        from aiter.ops.shuffle import shuffle_weight
    except Exception:
        return 0

    count = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.bias is not None:
            continue
        w = module.weight
        if w.dtype != torch.bfloat16:
            continue
        if w.ndim != 2:
            continue
        n, k = w.shape
        if n % require_multiple != 0 or k % require_multiple != 0:
            continue
        if hasattr(module, "_aiter_preshuffled_weight"):
            continue
        try:
            w_shuf = shuffle_weight(w, layout=layout)
            module._aiter_preshuffled_weight = w_shuf  # type: ignore[attr-defined]
            count += 1
        except Exception:
            # Conservative: if any layer can't be shuffled, skip it.
            continue
    return count


# ============================================================================
# Benchmark helper
# ============================================================================

def benchmark_ops():
    """Benchmark eager vs optimized (Triton) ops."""
    import time
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    print("=" * 60)
    print("OPTIMIZED OPS BENCHMARK (Triton)")
    print("=" * 60)
    print(f"TRITON_AVAILABLE: {TRITON_AVAILABLE}")
    
    # Test data
    batch, seq, hidden = 8, 512, 2048
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=device)
    weight = torch.ones(hidden, dtype=dtype, device=device)
    
    def benchmark(fn, name, warmup=10, iters=100):
        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iters):
            _ = fn()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters * 1000
        print(f"  {name}: {elapsed:.3f} ms")
        return elapsed
    
    # RMSNorm benchmark
    print("\nRMSNorm:")
    eager_time = benchmark(lambda: rms_norm_eager(x, weight), "eager")
    if TRITON_AVAILABLE:
        triton_time = benchmark(lambda: rms_norm_optimized(x, weight), "triton")
        print(f"  speedup: {eager_time/triton_time:.2f}x")
    
    # GELU+Mul benchmark
    print("\nGELU+Mul:")
    x2 = torch.randn(batch, seq, hidden * 2, dtype=dtype, device=device)
    eager_time = benchmark(lambda: gelu_tanh_and_mul_eager(x2), "eager")
    if TRITON_AVAILABLE:
        triton_time = benchmark(lambda: gelu_tanh_and_mul_optimized(x2), "triton")
        print(f"  speedup: {eager_time/triton_time:.2f}x")


if __name__ == "__main__":
    benchmark_ops()
