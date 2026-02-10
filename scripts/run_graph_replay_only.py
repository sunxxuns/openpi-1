#!/usr/bin/env python3
"""Minimal script: warm up, capture CUDAGraph, replay once (for rocprof tracing)."""
import os, sys, pathlib, shutil
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

os.environ.setdefault("USE_AITER_ATTENTION", "1")
os.environ.setdefault("USE_FUSED_PROJECTIONS", "1")
os.environ.setdefault("USE_AITER_GEMM", "1")
os.environ.setdefault("USE_OPTIMIZED_OPS", "1")
os.environ.setdefault("AITER_MASK_OVERRIDE", "1")
os.environ.setdefault("AITER_EXPERT_MASK_TYPE", "eager")
os.environ.setdefault("OPENPI_INDUCTOR_LOG", "0")
os.environ.setdefault("OPENPI_INDUCTOR_MEMORY_PLANNING", "0")
os.environ.setdefault("OPENPI_DISABLE_COMPILE_AITER_ATTN", "0")
os.environ.setdefault("OPENPI_AITER_ATTN_DIRECT_MHA", "1")
os.environ.setdefault("AUTO_PATCH_TRANSFORMERS", "1")
os.environ.setdefault("OPENPI_MANUAL_CUDAGRAPH", "0")
os.environ.setdefault("AITER_PRESHUFFLE_WEIGHTS", "0")
os.environ.setdefault("OPENPI_SKIP_MASKED_IMAGES", "0")

def _maybe_extend():
    if os.environ.get("AITER_CONFIG_GEMM_BF16"): return
    repo = pathlib.Path(__file__).resolve().parents[1]
    import torch
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except: arch = ""
    cfg = repo / "configs" / ("openpi_bf16_tuned_gemm_mi300.csv" if arch.startswith("gfx942") else "openpi_bf16_tuned_gemm.csv")
    if not cfg.exists(): cfg = repo / "configs" / "openpi_bf16_tuned_gemm.csv"
    if not cfg.exists(): return
    aiter_pkg = None
    try:
        import importlib.util as iu
        spec = iu.find_spec("aiter")
        if spec and spec.submodule_search_locations:
            aiter_pkg = pathlib.Path(list(spec.submodule_search_locations)[0])
    except: pass
    if not aiter_pkg:
        os.environ["AITER_CONFIG_GEMM_BF16"] = str(cfg); return
    paths = []
    d = aiter_pkg / "configs" / "bf16_tuned_gemm.csv"
    if d.exists(): paths.append(str(d))
    md = aiter_pkg / "configs" / "model_configs"
    if md.is_dir():
        for p in sorted(md.glob("*bf16_tuned_gemm*.csv")):
            if "untuned" not in p.name: paths.append(str(p))
    paths.append(str(cfg))
    os.environ["AITER_CONFIG_GEMM_BF16"] = os.pathsep.join(paths)

def _patch():
    if os.environ.get("AUTO_PATCH_TRANSFORMERS","0")!="1": return
    try:
        import transformers
        src = pathlib.Path(__file__).resolve().parents[1]/"src"/"openpi"/"models_pytorch"/"transformers_replace"/"models"
        dest = pathlib.Path(transformers.__file__).resolve().parent/"models"
        for c in src.iterdir():
            if c.is_dir(): shutil.copytree(c, dest/c.name, dirs_exist_ok=True)
    except: pass

_maybe_extend(); _patch()
import torch
from transformers.models.gemma.modeling_gemma import set_use_aiter_attention
set_use_aiter_attention(True)

def main():
    gpu_id = int(os.environ.get("OPENPI_GPU_ID", "7"))
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    from dataclasses import dataclass
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    @dataclass
    class Cfg:
        action_dim: int = 32; action_horizon: int = 10; max_token_len: int = 48
        dtype: str = "bfloat16"; paligemma_variant: str = "gemma_2b"
        action_expert_variant: str = "gemma_300m"; pi05: bool = False
    model = PI0Pytorch(Cfg()).to(device)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    if os.environ.get("USE_AITER_GEMM","0")=="1":
        try:
            from openpi.models_pytorch.aiter_ops import set_use_aiter_gemm, patch_linear_forward, AITER_GEMM_AVAILABLE
            if AITER_GEMM_AVAILABLE: set_use_aiter_gemm(True); patch_linear_forward()
        except: pass
    try: model.paligemma_with_expert.fuse_projections(verbose=False)
    except: pass
    if os.environ.get("OPENPI_FUSE_SIGLIP_QKV","0")=="1":
        try:
            from transformers.models.siglip.modeling_siglip import fuse_siglip_qkv_projections
            fuse_siglip_qkv_projections(model, verbose=False)
        except: pass
    if os.environ.get("AITER_MASK_OVERRIDE","0")=="1":
        try:
            em = os.environ.get("AITER_EXPERT_MASK_TYPE","eager")
            for l in model.paligemma_with_expert.paligemma.language_model.layers: l.self_attn._aiter_mask_type="full"
            for l in model.paligemma_with_expert.gemma_expert.model.layers: l.self_attn._aiter_mask_type=em
        except: pass
    model.eval()

    class Obs:
        def __init__(self, **kw):
            for k,v in kw.items(): setattr(self,k,v)
    obs = Obs(
        images={
            'base_0_rgb': torch.rand(1,3,224,224,dtype=torch.float32,device=device)*2-1,
            'left_wrist_0_rgb': torch.rand(1,3,224,224,dtype=torch.float32,device=device)*2-1,
            'right_wrist_0_rgb': torch.zeros(1,3,224,224,dtype=torch.float32,device=device),
        },
        image_masks={
            'base_0_rgb': torch.ones(1,dtype=torch.bool,device=device),
            'left_wrist_0_rgb': torch.ones(1,dtype=torch.bool,device=device),
            'right_wrist_0_rgb': torch.zeros(1,dtype=torch.bool,device=device),
        },
        state=torch.randn(1,32,dtype=torch.bfloat16,device=device),
        tokenized_prompt=torch.randint(0,256000,(1,20),dtype=torch.long,device=device),
        tokenized_prompt_mask=torch.ones(1,20,dtype=torch.bool,device=device),
        token_ar_mask=torch.ones(1,20,dtype=torch.int32,device=device),
        token_loss_mask=torch.zeros(1,20,dtype=torch.bool,device=device),
    )

    # Warmup
    print("Warmup...", flush=True)
    for _ in range(10):
        with torch.no_grad():
            model.sample_actions(device, obs, num_steps=10)
    torch.cuda.synchronize()

    # Capture graph
    try:
        from openpi.models_pytorch.rocm_cudagraph_dynamo_patch import patch_dynamo_preserve_global_state_for_rocm_cudagraph_capture
        patch_dynamo_preserve_global_state_for_rocm_cudagraph_capture()
    except: pass
    pool = torch.cuda.graphs.graph_pool_handle()
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad():
        for _ in range(5):
            model.sample_actions(device, obs, num_steps=10)
    torch.cuda.synchronize()
    print("Capturing graph...", flush=True)
    with torch.cuda.graph(graph, pool=pool):
        static_actions = model.sample_actions(device, obs, num_steps=10)
    torch.cuda.synchronize()
    print("Graph captured.", flush=True)

    # Pre-replay warmup
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    # Marker for rocprof
    print("ROCPROF_START", flush=True)
    graph.replay()
    torch.cuda.synchronize()
    print("ROCPROF_END", flush=True)

    import time
    t0 = time.perf_counter()
    graph.replay()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"Graph replay: {(t1-t0)*1000:.1f} ms", flush=True)

if __name__ == "__main__":
    main()
