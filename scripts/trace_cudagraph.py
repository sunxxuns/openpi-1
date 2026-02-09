#!/usr/bin/env python3
"""Trace CUDAGraph replay with torch profiler using minimal overhead.
Only profiles the graph replay (not compilation/warmup), so the trace
reflects the actual ~24ms execution path.
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("AMD_LOG_LEVEL", "0")
os.environ.setdefault("OPENPI_MANUAL_CUDAGRAPH", "1")
os.environ.setdefault("OPENPI_BATCH_SIGLIP", "1")
os.environ.setdefault("OPENPI_FUSE_SIGLIP_QKV", "1")
os.environ.setdefault("OPENPI_EAGER_ATTN_USE_SDPA", "1")
os.environ.setdefault("AITER_PRESHUFFLE_WEIGHTS", "0")
os.environ.setdefault("USE_AITER_ATTENTION", "1")
os.environ.setdefault("USE_FUSED_PROJECTIONS", "1")
os.environ.setdefault("USE_AITER_GEMM", "1")
os.environ.setdefault("AITER_MASK_OVERRIDE", "1")
os.environ.setdefault("OPENPI_ROUTE_FUSED_LINEAR_TO_AITER", "1")
os.environ.setdefault("OPENPI_ROUTE_SIGLIP_FUSED_QKV_TO_AITER", "1")
os.environ.setdefault("OPENPI_ROUTE_FUSED_LINEAR_M_THRESH", "1000000")
os.environ.setdefault("OPENPI_COORD_DESCENT", "1")
os.environ.setdefault("OPENPI_BENCHMARK_KERNEL", "1")
os.environ.setdefault("OPENPI_GROUP_FUSION", "1")
os.environ.setdefault("OPENPI_MAX_FUSION_SIZE", "128")
os.environ.setdefault("OPENPI_FREEZING", "1")

import torch
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

GPU = int(os.environ.get("OPENPI_GPU_ID", "7"))
dev = torch.device(f"cuda:{GPU}")
torch.cuda.set_device(dev)

@dataclass
class C:
    action_dim:int=32; action_horizon:int=10; max_token_len:int=48
    dtype:str='bfloat16'; paligemma_variant:str='gemma_2b'
    action_expert_variant:str='gemma_300m'; pi05:bool=os.environ.get("OPENPI_PI05","0")=="1"

print("Building model...")
model = PI0Pytorch(C()).to(dev).eval()
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
try: model.paligemma_with_expert.fuse_projections(verbose=False)
except: pass
try:
    from transformers.models.siglip.modeling_siglip import fuse_siglip_qkv_projections
    fuse_siglip_qkv_projections(model, verbose=False)
except: pass

# Apply mask overrides (skip torch.unique mask check during graph capture)
for layer in model.paligemma_with_expert.paligemma.language_model.layers:
    layer.self_attn._aiter_mask_type = "full"
for layer in model.paligemma_with_expert.gemma_expert.model.layers:
    layer.self_attn._aiter_mask_type = "eager"

B = 1
class Obs:
    def __init__(self, **kw):
        for k,v in kw.items(): setattr(self, k, v)
obs = Obs(
    images={'base_0_rgb':torch.rand(B,3,224,224,device=dev)*2-1,
            'left_wrist_0_rgb':torch.rand(B,3,224,224,device=dev)*2-1,
            'right_wrist_0_rgb':torch.rand(B,3,224,224,device=dev)*2-1},
    image_masks={'base_0_rgb':torch.ones(B,dtype=torch.bool,device=dev),
                 'left_wrist_0_rgb':torch.ones(B,dtype=torch.bool,device=dev),
                 'right_wrist_0_rgb':torch.ones(B,dtype=torch.bool,device=dev)},
    state=torch.randn(B,32,dtype=torch.bfloat16,device=dev),
    tokenized_prompt=torch.randint(0,256000,(B,20),dtype=torch.long,device=dev),
    tokenized_prompt_mask=torch.ones(B,20,dtype=torch.bool,device=dev),
    token_ar_mask=torch.ones(B,20,dtype=torch.int32,device=dev),
    token_loss_mask=torch.zeros(B,20,dtype=torch.bool,device=dev),
)

# Full warmup + compile
print("Warmup + compile...")
with torch.no_grad():
    for _ in range(10):
        model.sample_actions(dev, obs, num_steps=10)
torch.cuda.synchronize()

# Capture CUDAGraph
print("Capturing CUDAGraph...")
try:
    from openpi.models_pytorch.rocm_cudagraph_dynamo_patch import \
        patch_dynamo_preserve_global_state_for_rocm_cudagraph_capture
    patch_dynamo_preserve_global_state_for_rocm_cudagraph_capture()
except: pass

pool = torch.cuda.graphs.graph_pool_handle()
graph = torch.cuda.CUDAGraph()
with torch.no_grad():
    for _ in range(5):
        model.sample_actions(dev, obs, num_steps=10)
torch.cuda.synchronize()
with torch.cuda.graph(graph, pool=pool):
    static_out = model.sample_actions(dev, obs, num_steps=10)
torch.cuda.synchronize()

# Warmup replay
for _ in range(20):
    graph.replay()
torch.cuda.synchronize()

# Verify latency
s = torch.cuda.Event(enable_timing=True)
e = torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(20):
    graph.replay()
e.record(); e.synchronize()
print(f"Graph replay: {s.elapsed_time(e)/20:.1f} ms")

# Profile ONLY the graph replay — single iteration
print("Profiling 1 graph replay...")
trace_dir = os.environ.get("PROFILE_DIR", "traces")
os.makedirs(trace_dir, exist_ok=True)
trace_path = f"{trace_dir}/mi350_cudagraph_replay.json"

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
) as prof:
    graph.replay()
    torch.cuda.synchronize()

prof.export_chrome_trace(trace_path)
sz = os.path.getsize(trace_path) / 1024
print(f"Trace: {trace_path} ({sz:.0f} KB)")
print()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
