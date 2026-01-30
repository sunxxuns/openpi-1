import logging
import math
import os
import time

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models_pytorch.gemma_config_pytorch as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def _maybe_enable_inductor_compile_logging():
    """Enable detailed inductor logging for debugging torch.compile on AMD MI350.
    
    Set OPENPI_INDUCTOR_LOG=1 to enable basic logging.
    Set OPENPI_INDUCTOR_LOG=2 for verbose logging (includes graph structure).
    """
    log_level = os.environ.get("OPENPI_INDUCTOR_LOG", "0")
    if log_level == "0":
        return
    
    verbose = log_level == "2"
    
    try:
        import torch._inductor.compile_fx as compile_fx
        if getattr(compile_fx, "_openpi_wrapped", False):
            return
        original_compile_fx = compile_fx.compile_fx
        
        # Track compilation stats
        compile_stats = {"count": 0, "total_time": 0.0, "graphs": []}

        def wrapped_compile_fx(gm, example_inputs, *args, **kwargs):
            nonlocal compile_stats
            compile_stats["count"] += 1
            graph_id = compile_stats["count"]
            
            name = getattr(gm, "name", getattr(gm, "_name", gm.__class__.__name__))
            node_count = len(list(gm.graph.nodes)) if hasattr(gm, "graph") else "?"
            
            # Get input shapes for debugging
            input_shapes = []
            try:
                for inp in example_inputs:
                    if hasattr(inp, "shape"):
                        input_shapes.append(tuple(inp.shape))
                    else:
                        input_shapes.append(type(inp).__name__)
            except Exception:
                input_shapes = ["?"]
            
            print(
                f"[inductor] #{graph_id} compile_fx START name={name} nodes={node_count} inputs={input_shapes}",
                flush=True,
            )
            
            # Log graph structure in verbose mode
            if verbose and hasattr(gm, "graph"):
                print(f"[inductor] #{graph_id} graph ops:", flush=True)
                op_counts = {}
                for node in gm.graph.nodes:
                    op = f"{node.op}:{node.target}" if node.op == "call_function" else node.op
                    op_counts[op] = op_counts.get(op, 0) + 1
                for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:10]:
                    print(f"  {op}: {count}", flush=True)
            
            start = time.perf_counter()
            try:
                result = original_compile_fx(gm, example_inputs, *args, **kwargs)
                elapsed = time.perf_counter() - start
                compile_stats["total_time"] += elapsed
                compile_stats["graphs"].append({"name": name, "nodes": node_count, "time": elapsed})
                print(
                    f"[inductor] #{graph_id} compile_fx END name={name} elapsed={elapsed:.1f}s (total: {compile_stats['total_time']:.1f}s)",
                    flush=True,
                )
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                print(
                    f"[inductor] #{graph_id} compile_fx FAILED name={name} elapsed={elapsed:.1f}s error={exc}",
                    flush=True,
                )
                raise

        compile_fx.compile_fx = wrapped_compile_fx
        compile_fx._openpi_wrapped = True
        print(f"[inductor] compile_fx logging enabled (verbose={verbose})", flush=True)
        
        # Also enable graph break logging (if available)
        try:
            import torch._dynamo.config as dynamo_config
            if hasattr(dynamo_config, "log_graph_breaks"):
                dynamo_config.log_graph_breaks = True
                print("[inductor] graph break logging enabled", flush=True)
        except Exception:
            pass
            
    except Exception as exc:
        print(f"[inductor] failed to enable compile logging: {exc}", flush=True)


def _configure_inductor_for_amd():
    """Configure torch inductor for AMD MI350 (ROCm).

    This project previously experimented with enabling Inductor cudagraphs and forcing
    Triton GEMM to reduce overhead. On MI350, benchmarks showed:
    - **rocBLAS (ATen GEMM) is typically 35-55% faster than Triton GEMM**
    - Inductor/graph-based "reduce-overhead" paths can be unstable on ROCm depending
      on what libraries are used (e.g. capture limitations).

    So the default here is MI350-safe:
    - GEMMs: prefer ATen/rocBLAS
    - Elementwise: allow Triton fusion/epilogues
    - Inductor cudagraphs: disabled by default (can be re-enabled via env)
    """
    try:
        import torch._inductor.config as inductor_config
        import torch._dynamo.config as dynamo_config
        
        # Check if we're on ROCm
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        
        if is_rocm:
            rocm_version = torch.version.hip or "0.0"
            print(f"[inductor] Detected ROCm {rocm_version}, applying MI350-safe config", flush=True)

            # -----------------------------------------------------------------
            # Graphs (Inductor-level): OFF by default on ROCm
            # -----------------------------------------------------------------
            # If you want to experiment, set OPENPI_ENABLE_INDUCTOR_CUDAGRAPHS=1
            enable_graphs = os.environ.get("OPENPI_ENABLE_INDUCTOR_CUDAGRAPHS", "0") == "1"
            inductor_config.triton.cudagraphs = bool(enable_graphs)
            inductor_config.triton.cudagraph_trees = bool(enable_graphs)
            if enable_graphs and hasattr(inductor_config.triton, "cudagraph_skip_autotuning"):
                inductor_config.triton.cudagraph_skip_autotuning = True

            # -----------------------------------------------------------------
            # GEMM: prefer ATen/rocBLAS
            # -----------------------------------------------------------------
            inductor_config.max_autotune_gemm_backends = "ATEN"
            inductor_config.max_autotune = False
            inductor_config.coordinate_descent_tuning = False

            # -----------------------------------------------------------------
            # Fusion: keep the good defaults (avoid over-fusing huge graphs)
            # -----------------------------------------------------------------
            inductor_config.epilogue_fusion = True
            inductor_config.pattern_matcher = True
            # Allow experiments: more fusion can reduce kernel count (helps hipGraphLaunch)
            # but may also regress kernel quality on ROCm. Default remains conservative.
            inductor_config.aggressive_fusion = os.environ.get("OPENPI_AGGRESSIVE_FUSION", "0") == "1"
            if hasattr(inductor_config.triton, "multi_kernel"):
                # multi_kernel>1 can reduce launch count for some patterns; keep default=1
                inductor_config.triton.multi_kernel = int(os.environ.get("OPENPI_TRITON_MULTI_KERNEL", "1"))

            # -----------------------------------------------------------------
            # Memory planning
            # -----------------------------------------------------------------
            if hasattr(inductor_config, "reorder_for_locality"):
                inductor_config.reorder_for_locality = True
            if hasattr(inductor_config, "memory_planning"):
                # On ROCm, Inductor's memory_planning can hit deep recursion for very
                # large graphs (seen when compiling through attention). Keep enabled
                # disabled by default for stability (can be re-enabled via env).
                inductor_config.memory_planning = os.environ.get(
                    "OPENPI_INDUCTOR_MEMORY_PLANNING", "0"
                ) == "1"

            # -----------------------------------------------------------------
            # Dynamo cache / compilation overhead
            # -----------------------------------------------------------------
            dynamo_config.cache_size_limit = max(getattr(dynamo_config, "cache_size_limit", 64), 128)
            dynamo_config.suppress_errors = False

            # -----------------------------------------------------------------
            # HIP runtime knobs
            # -----------------------------------------------------------------
            os.environ.setdefault("HIP_LAUNCH_BLOCKING", "0")
            os.environ.setdefault("AMD_LOG_LEVEL", "0")
            os.environ.setdefault("HIP_CACHE_ENABLED", "1")

            print("[inductor] MI350 config applied:", flush=True)
            print(f"  - Inductor cudagraphs: {inductor_config.triton.cudagraphs}", flush=True)
            print("  - GEMM backend: ATEN (rocBLAS)", flush=True)
            print(f"  - aggressive_fusion: {inductor_config.aggressive_fusion}", flush=True)
            if hasattr(inductor_config, "memory_planning"):
                print(f"  - memory_planning: {inductor_config.memory_planning}", flush=True)
        else:
            print("[inductor] CUDA detected, using default optimizations", flush=True)
            
    except Exception as exc:
        print(f"[inductor] failed to configure AMD optimizations: {exc}", flush=True)


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        _maybe_enable_inductor_compile_logging()
        torch.set_float32_matmul_precision("high")
        
        # Configure dynamo to handle dynamic KV cache shapes
        import torch._dynamo.config as dynamo_config
        dynamo_config.cache_size_limit = 64  # Increase from default 8
        dynamo_config.suppress_errors = False  # Surface errors for debugging
        
        # Apply AMD-specific inductor optimizations
        _configure_inductor_for_amd()
        
        # Additional inductor config (can be overridden by env vars)
        import torch._inductor.config as inductor_config
        
        # Optional: force Triton matmul for better fusion (longer compile)
        use_triton_mm = os.environ.get("USE_TRITON_MM", "0") == "1"
        if use_triton_mm:
            inductor_config.max_autotune = True
            inductor_config.coordinate_descent_tuning = False
            inductor_config.max_autotune_gemm_backends = "TRITON"
        
        # Apply torch.compile with appropriate mode
        # - "reduce-overhead": Uses CUDA graphs (faster on both CUDA and ROCm)
        # - "default": Standard compilation (slower but more stable)
        # - "max-autotune": Most optimized but slowest compile
        # - "disable": Skip torch.compile entirely
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        # On ROCm, "reduce-overhead" is frequently a footgun (graph capture limitations).
        # Default to "default" unless the user explicitly overrides via env.
        default_mode = "default" if is_rocm else "reduce-overhead"
        compile_mode = os.environ.get("TORCH_COMPILE_MODE", default_mode)
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        if is_rocm:
            print(f"[torch.compile] ROCm detected, mode={compile_mode}", flush=True)
        if compile_mode != "disable":
            self.sample_actions = torch.compile(self.sample_actions, mode=compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Check for transformers_replace installation (can be skipped with SKIP_TRANSFORMERS_CHECK=1)
        skip_check = os.environ.get("SKIP_TRANSFORMERS_CHECK", "0") == "1"
        if not skip_check:
            msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
            try:
                from transformers.models.siglip import check

                if not check.check_whether_transformers_replace_is_installed_correctly():
                    logging.warning(msg)
            except ImportError:
                logging.warning(msg)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        # CUDAGraph-capture friendly: avoid creating new tensors during capture.
        # Use Python scalar for dt and cache the timestep schedule tensor.
        times_key = (str(device), int(num_steps))
        times = getattr(self, "_openpi_sample_actions_times", None)
        if not isinstance(times, dict) or times.get("key") != times_key:
            # Construct once (outside capture) then reuse.
            # Note: this allocates; safe as long as first call happens before capture.
            t0 = 1.0
            # times[step] = 1.0 + step*dt
            t = torch.arange(num_steps, device=device, dtype=torch.float32) * float(dt) + float(t0)
            times = {"key": times_key, "t": t}
            setattr(self, "_openpi_sample_actions_times", times)
        t_schedule = times["t"]

        x_t = noise
        # Use for loop instead of while to avoid torch.compile recursion issues
        # The while loop `while time >= -dt / 2` runs exactly num_steps iterations
        for step in range(num_steps):
            # Use cached scalar timestep tensor; expand is a view.
            expanded_time = t_schedule[step].expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + (dt * v_t)
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
