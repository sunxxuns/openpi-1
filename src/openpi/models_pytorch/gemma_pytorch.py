from typing import Literal

import os
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma
from transformers.models.gemma.modeling_gemma import get_use_aiter_attention, aiter_attention_forward

try:
    # Optional: route fused F.linear GEMMs through aiter tuned GEMM.
    from openpi.models_pytorch.aiter_ops import get_use_aiter_gemm, aiter_linear
except Exception:
    def get_use_aiter_gemm() -> bool:  # type: ignore[override]
        return False

    def aiter_linear(x, weight, bias=None):  # type: ignore[override]
        return F.linear(x, weight, bias)


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    attn = layer.self_attn

                    # Use fused QKV projection when available (reduces 3 GEMMs to 1)
                    if hasattr(attn, "_use_fused_qkv") and attn._use_fused_qkv and hasattr(attn, "_fused_qkv_weight"):
                        q_size = attn.q_proj.weight.shape[0]
                        k_size = attn.k_proj.weight.shape[0]
                        # Optional MI350-only experiment:
                        # Routing fused GEMMs through aiter can help for *small-M* (decode-ish) shapes,
                        # but can regress for large M (prefill-ish) shapes. Keep default behavior as F.linear.
                        route_fused = os.environ.get("OPENPI_ROUTE_FUSED_LINEAR_TO_AITER", "0") == "1"
                        if route_fused and get_use_aiter_gemm():
                            try:
                                # hidden_states is typically [B, S, K]
                                m = int(hidden_states.shape[0]) * int(hidden_states.shape[1]) if hidden_states.ndim == 3 else int(hidden_states.shape[0])
                            except Exception:
                                m = 0
                            m_thresh = int(os.environ.get("OPENPI_ROUTE_FUSED_LINEAR_M_THRESH", "64"))
                            if m and m <= m_thresh:
                                fused_w = getattr(attn, "_aiter_preshuffled_fused_qkv_weight", attn._fused_qkv_weight)
                                qkv_states = aiter_linear(hidden_states, fused_w, bias=None)
                            else:
                                qkv_states = F.linear(hidden_states, attn._fused_qkv_weight)
                        else:
                            qkv_states = F.linear(hidden_states, attn._fused_qkv_weight)
                        query_state = qkv_states[..., :q_size].view(hidden_shape).transpose(1, 2)
                        key_state = qkv_states[..., q_size:q_size + k_size].view(hidden_shape).transpose(1, 2)
                        value_state = qkv_states[..., q_size + k_size:].view(hidden_shape).transpose(1, 2)
                    else:
                        query_state = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        key_state = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        value_state = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation - use aiter if enabled, otherwise eager
                if get_use_aiter_attention():
                    att_output, _ = aiter_attention_forward(
                        self.paligemma.language_model.layers[layer_idx].self_attn,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        scaling,
                    )
                else:
                    att_output, _ = modeling_gemma.eager_attention_forward(
                        self.paligemma.language_model.layers[layer_idx].self_attn,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        scaling,
                    )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values

    def fuse_projections(self, fuse_qkv: bool = True, fuse_gate_up: bool = True, verbose: bool = True):
        """Fuse linear projections to reduce kernel launch overhead.

        - QKV fusion: 3 GEMMs -> 1 GEMM
        - Gate+Up fusion: 2 GEMMs -> 1 GEMM
        """
        if os.environ.get("USE_FUSED_PROJECTIONS", "0") != "1":
            if verbose:
                print("Projection fusion disabled. Set USE_FUSED_PROJECTIONS=1 to enable.")
            return

        qkv_count = 0
        gate_up_count = 0

        # Optional fused activation hook
        gelu_tanh_and_mul = None
        try:
            from openpi.models_pytorch.aiter_ops import gelu_tanh_and_mul as _gelu_tanh_and_mul

            gelu_tanh_and_mul = _gelu_tanh_and_mul
        except Exception:
            gelu_tanh_and_mul = None

        models = [self.paligemma.language_model, self.gemma_expert.model]
        for model in models:
            if not hasattr(model, "layers"):
                continue
            for layer in model.layers:
                if fuse_qkv and hasattr(layer, "self_attn"):
                    attn = layer.self_attn
                    if hasattr(attn, "q_proj") and hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
                        if not hasattr(attn, "_use_fused_qkv") or not attn._use_fused_qkv:
                            fused_weight = torch.cat(
                                [attn.q_proj.weight.data, attn.k_proj.weight.data, attn.v_proj.weight.data], dim=0
                            )
                            attn.register_buffer("_fused_qkv_weight", fused_weight)
                            attn._use_fused_qkv = True
                            qkv_count += 1
                            # Optional: preshuffle fused QKV weights for aiter asm GEMM paths.
                            if (
                                os.environ.get("AITER_PRESHUFFLE_WEIGHTS", "0") == "1"
                                and os.environ.get("OPENPI_PRESHUFFLE_FUSED_WEIGHTS", "0") == "1"
                            ):
                                try:
                                    from aiter.ops.shuffle import shuffle_weight
                                    require_multiple = int(os.environ.get("AITER_PRESHUFFLE_REQUIRE_MULTIPLE", "256"))
                                    n, k = fused_weight.shape
                                    if (n % require_multiple == 0) and (k % require_multiple == 0):
                                        w_shuf = shuffle_weight(fused_weight, layout=(16, 16))
                                        attn.register_buffer("_aiter_preshuffled_fused_qkv_weight", w_shuf)
                                except Exception:
                                    pass

                if fuse_gate_up and hasattr(layer, "mlp"):
                    mlp = layer.mlp
                    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
                        if not hasattr(mlp, "_use_fused") or not mlp._use_fused:
                            fused_weight = torch.cat(
                                [mlp.gate_proj.weight.data, mlp.up_proj.weight.data], dim=0
                            )
                            mlp.register_buffer("_fused_gate_up_weight", fused_weight)
                            mlp._use_fused = True
                            if gelu_tanh_and_mul is not None:
                                mlp._gelu_tanh_and_mul = gelu_tanh_and_mul
                            gate_up_count += 1
                            # Optional: preshuffle fused gate+up weights for aiter asm GEMM paths.
                            if (
                                os.environ.get("AITER_PRESHUFFLE_WEIGHTS", "0") == "1"
                                and os.environ.get("OPENPI_PRESHUFFLE_FUSED_WEIGHTS", "0") == "1"
                            ):
                                try:
                                    from aiter.ops.shuffle import shuffle_weight
                                    require_multiple = int(os.environ.get("AITER_PRESHUFFLE_REQUIRE_MULTIPLE", "256"))
                                    n, k = fused_weight.shape
                                    if (n % require_multiple == 0) and (k % require_multiple == 0):
                                        w_shuf = shuffle_weight(fused_weight, layout=(16, 16))
                                        mlp.register_buffer("_aiter_preshuffled_fused_gate_up_weight", w_shuf)
                                except Exception:
                                    pass

        if verbose:
            print(f"Fused {qkv_count} QKV projections, {gate_up_count} Gate+Up projections")
