"""GEMM fusion utilities for model optimization."""

import torch
import torch.nn.functional as F
from types import MethodType


def fuse_qkv_projections(model):
    """
    Fuse Q, K, V projections into a single GEMM for GemmaAttention layers.
    
    This reduces 3 kernel launches to 1 per layer.
    For GQA, Q has different num_heads than K/V.
    """
    for name, module in model.named_modules():
        # Check if this is a GemmaAttention module with q/k/v projections
        # Make sure we don't fuse SigLIP attention which has different signature
        if (hasattr(module, 'q_proj') and 
            hasattr(module, 'k_proj') and 
            hasattr(module, 'v_proj') and
            hasattr(module, 'o_proj') and  # Also check for o_proj to confirm it's attention
            hasattr(module, 'head_dim')):   # head_dim is specific to GemmaAttention
            
            # Skip if already fused
            if hasattr(module, '_fused_qkv_weight'):
                continue
                
            q_proj = module.q_proj
            k_proj = module.k_proj
            v_proj = module.v_proj
            
            # Get dimensions
            hidden_size = q_proj.in_features
            q_out_features = q_proj.out_features
            k_out_features = k_proj.out_features
            v_out_features = v_proj.out_features
            
            # Fuse weights: [q_out + k_out + v_out, hidden_size]
            fused_weight = torch.cat([
                q_proj.weight.data,
                k_proj.weight.data,
                v_proj.weight.data,
            ], dim=0)
            
            # Fuse biases if they exist
            has_bias = q_proj.bias is not None
            fused_bias = None
            if has_bias:
                fused_bias = torch.cat([
                    q_proj.bias.data,
                    k_proj.bias.data,
                    v_proj.bias.data,
                ], dim=0)
            
            # Register fused parameters as buffers
            module.register_buffer('_fused_qkv_weight', fused_weight)
            if fused_bias is not None:
                module.register_buffer('_fused_qkv_bias', fused_bias)
            
            # Store output dimensions for splitting
            module._qkv_split_sizes = [q_out_features, k_out_features, v_out_features]
            
            # Replace forward method with fused version using MethodType
            module._original_forward = module.forward
            module.forward = MethodType(fused_qkv_forward, module)
            
    return model


def fused_qkv_forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None, cache_position=None, use_cache=False, **kwargs):
    """Fused QKV forward pass."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb
    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    
    # Single fused GEMM
    if hasattr(self, '_fused_qkv_bias'):
        qkv = F.linear(hidden_states, self._fused_qkv_weight, self._fused_qkv_bias)
    else:
        qkv = F.linear(hidden_states, self._fused_qkv_weight)
    
    # Split back into Q, K, V
    q_out_dim, k_out_dim, v_out_dim = self._qkv_split_sizes
    query_states, key_states, value_states = qkv.split([q_out_dim, k_out_dim, v_out_dim], dim=-1)
    
    # Reshape for attention
    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
    
    # Apply RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # Use cache if provided
    if past_key_value is not None:
        if use_cache:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            key_states = torch.cat([past_key_value[self.layer_idx][0], key_states], dim=2)
            value_states = torch.cat([past_key_value[self.layer_idx][1], value_states], dim=2)
    
    # Call attention implementation
    attn_impl = self.config._attn_implementation if hasattr(self.config, '_attn_implementation') else 'eager'
    attention_interface = ALL_ATTENTION_FUNCTIONS.get(attn_impl, None)
    
    if attention_interface is None:
        # Fallback to eager
        attention_interface = eager_attention_forward
    
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Eager attention without SDPA."""
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling
    
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, None


def fuse_gate_up_projections(model):
    """
    Fuse gate and up projections into a single GEMM for GemmaMLP layers.
    
    This reduces 2 kernel launches to 1 per layer.
    """
    for name, module in model.named_modules():
        # Check if this is a GemmaMLP module with gate/up/down projections
        if (hasattr(module, 'gate_proj') and 
            hasattr(module, 'up_proj') and
            hasattr(module, 'down_proj') and
            hasattr(module, 'act_fn')):
            
            # Skip if already fused
            if hasattr(module, '_fused_gate_up_weight'):
                continue
                
            gate_proj = module.gate_proj
            up_proj = module.up_proj
            
            # Get dimensions
            hidden_size = gate_proj.in_features
            intermediate_size = gate_proj.out_features
            
            # Fuse weights: [2 * intermediate_size, hidden_size]
            fused_weight = torch.cat([
                gate_proj.weight.data,
                up_proj.weight.data,
            ], dim=0)
            
            # Fuse biases if they exist
            has_bias = gate_proj.bias is not None
            fused_bias = None
            if has_bias:
                fused_bias = torch.cat([
                    gate_proj.bias.data,
                    up_proj.bias.data,
                ], dim=0)
            
            # Register fused parameters as buffers
            module.register_buffer('_fused_gate_up_weight', fused_weight)
            if fused_bias is not None:
                module.register_buffer('_fused_gate_up_bias', fused_bias)
            
            # Replace forward method with fused version using MethodType
            module._original_forward = module.forward
            module.forward = MethodType(fused_mlp_forward, module)
            
    return model


def fused_mlp_forward(self, x):
    """Fused Gate+Up forward pass."""
    # Single fused GEMM
    if hasattr(self, '_fused_gate_up_bias'):
        gate_up = F.linear(x, self._fused_gate_up_weight, self._fused_gate_up_bias)
    else:
        gate_up = F.linear(x, self._fused_gate_up_weight)
    
    # Split into gate and up
    gate, up = gate_up.chunk(2, dim=-1)
    
    # Apply activation and multiply
    down_proj = self.down_proj(self.act_fn(gate) * up)
    return down_proj