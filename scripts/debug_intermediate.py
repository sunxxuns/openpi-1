#!/usr/bin/env python3
"""
Debug intermediate values for H200 vs MI350 comparison.
Uses fixed seed for model initialization to ensure identical weights.
"""
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

# Set seed BEFORE any model imports to ensure identical weight initialization
torch.manual_seed(42)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ["TORCH_COMPILE_MODE"] = "disable"

from dataclasses import dataclass
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

@dataclass
class Pi0ConfigPytorch:
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 48
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    pi05: bool = False

device = torch.device("cuda:0")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")

inputs = torch.load("traces/numerical_check_inputs.pt", weights_only=False)
noise = inputs["noise"].to(device)

# Create model with seeded weights
torch.manual_seed(42)  # Reset seed before model creation
model = PI0Pytorch(Pi0ConfigPytorch())
model = model.to(device)
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
model.eval()

# Verify weight checksum
weight_sum = sum(p.sum().item() for p in model.parameters())
print(f"Weight checksum: {weight_sum:.2f}")

# Setup observation
class SimpleObservation:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

images = {
    'base_0_rgb': inputs['base_0_rgb'].to(device),
    'left_wrist_0_rgb': inputs['left_wrist_0_rgb'].to(device),
    'right_wrist_0_rgb': inputs['right_wrist_0_rgb'].to(device),
}
image_masks = {
    'base_0_rgb': inputs['image_mask_base_0_rgb'].to(device),
    'left_wrist_0_rgb': inputs['image_mask_left_wrist_0_rgb'].to(device),
    'right_wrist_0_rgb': inputs['image_mask_right_wrist_0_rgb'].to(device),
}

observation = SimpleObservation(
    images=images,
    image_masks=image_masks,
    state=inputs['state'].to(device),
    tokenized_prompt=inputs['tokenized_prompt'].to(device),
    tokenized_prompt_mask=inputs['tokenized_prompt_mask'].to(device),
    token_ar_mask=inputs['token_ar_mask'].to(device),
    token_loss_mask=inputs['token_loss_mask'].to(device),
)

print("\n" + "="*60)
print("INTERMEDIATE VALUES")
print("="*60)

with torch.inference_mode():
    # Step 1: Check prefix embeddings
    imgs, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(imgs, img_masks, lang_tokens, lang_masks)
    
    print(f"\n1. prefix_embs.sum() = {prefix_embs.sum().item():.2f}")
    print(f"1. prefix_embs[0,100,:3] = {[f'{v:.4f}' for v in prefix_embs[0,100,:3].tolist()]}")
    
    # Step 2: Check KV cache
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    
    kv_layer0_keys = past_key_values.key_cache[0]
    print(f"\n2. KV_layer0_keys.sum() = {kv_layer0_keys.sum().item():.2f}")
    
    # Step 3: Full inference with 1 step
    output_1step = model.sample_actions(device, observation, noise=noise.clone(), num_steps=1)
    print(f"\n3. sample_actions(steps=1).sum() = {output_1step.sum().item():.2f}")
    print(f"3. output[0,0,:5] = {[f'{v:.2f}' for v in output_1step[0,0,:5].tolist()]}")
    
    # Step 4: Full inference with 10 steps
    output_10step = model.sample_actions(device, observation, noise=noise.clone(), num_steps=10)
    print(f"\n4. sample_actions(steps=10).sum() = {output_10step.sum().item():.2f}")
    print(f"4. output[0,0,:5] = {[f'{v:.2f}' for v in output_10step[0,0,:5].tolist()]}")

print("\n" + "="*60)
