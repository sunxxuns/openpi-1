"""ROCm CUDAGraph capture helpers.

Why this exists:
- On ROCm, capturing a graph can fail if Dynamo (torch.compile) decides to trace
  a new frame while the stream is capturing.
- Dynamo's frame transform wrapper (`preserve_global_state`) saves CUDA RNG state
  (`torch.cuda.get_rng_state()`), which internally queries the generator seed.
- ROCm disallows this during capture and raises:
  `RuntimeError: Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture`

This module provides a best-effort runtime patch that makes Dynamo skip CUDA RNG
state preservation ONLY while a stream capture is active.

Scope/safety:
- This patch is applied only when explicitly requested (e.g. in benchmark code).
- It preserves CPU/Python RNG + torch CPU RNG state as before.
- Outside capture, behavior is unchanged.
"""

from __future__ import annotations

from typing import Any, Callable


def patch_dynamo_preserve_global_state_for_rocm_cudagraph_capture() -> bool:
    """Patch Dynamo to skip CUDA RNG get_state during stream capture.

    Returns:
        True if the patch was applied (or already applied), False otherwise.
    """
    try:
        import contextlib
        import functools
        import random as _py_random

        import torch
        import torch._dynamo.convert_frame as _cf

        if getattr(_cf, "_openpi_skip_cuda_rng_during_capture", False):
            return True

        def _preserve_global_state_skip_cuda_rng(fn: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(fn)
            def _fn(*args: Any, **kwargs: Any) -> Any:
                guards = _cf.GlobalStateGuard()
                prior_grad_mode = torch.is_grad_enabled()
                with (
                    torch._C._PreserveDispatchKeyGuard(),
                    _cf.maybe_disable_inference_mode(),
                    _cf.maybe_disable_inference_mode_for_fake_prop(),
                ):
                    prior_inference_mode = torch.is_inference_mode_enabled()
                    prior_deterministic = torch.are_deterministic_algorithms_enabled()
                    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
                    prior_mobile_allocator_state = torch._C._is_default_mobile_cpu_allocator_set()

                    py_rng_state = _py_random.getstate()
                    prior_dtype = torch.get_default_dtype()
                    torch_rng_state = torch.random.get_rng_state()

                    cuda_rng_state = None
                    if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
                        cuda_rng_state = torch.cuda.get_rng_state()

                    cuda_matmul_fp32_prec = torch._C._get_fp32_precision_getter("cuda", "matmul")
                    prior_fwd_from_src = torch.fx.graph_module._forward_from_src
                    torch.fx.graph_module._forward_from_src = _cf.fx_forward_from_src_skip_result

                    cleanup = _cf.setup_compile_debug()
                    exit_stack = contextlib.ExitStack()
                    exit_stack.enter_context(torch.fx._symbolic_trace._maybe_revert_all_patches())
                    exit_stack.enter_context(_cf.torch_function_mode_stack_state_mgr)
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        cleanup.close()
                        exit_stack.close()

                        torch._C._set_grad_enabled(prior_grad_mode)
                        torch.autograd.grad_mode._enter_inference_mode(prior_inference_mode)
                        torch.use_deterministic_algorithms(prior_deterministic, warn_only=prior_warn_only)
                        _py_random.setstate(py_rng_state)
                        torch.random.set_rng_state(torch_rng_state)
                        torch.set_default_dtype(prior_dtype)

                        curr_mobile_allocator_state = torch._C._is_default_mobile_cpu_allocator_set()
                        if prior_mobile_allocator_state != curr_mobile_allocator_state:
                            torch._C._unset_default_mobile_cpu_allocator()

                        if cuda_rng_state is not None:
                            torch.cuda.set_rng_state(cuda_rng_state)

                        torch._C._set_fp32_precision_setter("cuda", "matmul", cuda_matmul_fp32_prec)
                        torch.fx.graph_module._forward_from_src = prior_fwd_from_src
                        assert guards.check(), f"Global {guards.reason()}state changed while dynamo tracing"

            _fn._torchdynamo_orig_backend = fn  # type: ignore[attr-defined]
            return _fn

        _cf.preserve_global_state = _preserve_global_state_skip_cuda_rng  # type: ignore[assignment]
        _cf._openpi_skip_cuda_rng_during_capture = True
        return True
    except Exception:
        return False

