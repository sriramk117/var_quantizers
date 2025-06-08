#!/usr/bin/env python3
"""
Demo script for VAR (Visual AutoRegressive) model sampling
Converted from Jupyter notebook for standard Python execution

For an interactive experience, head over to the demo platform: https://var.vision/demo
"""

################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch
import torchvision
import random
import numpy as np
import PIL.Image as PImage
import PIL.ImageDraw as PImageDraw

from models.var import VAR
# Disable default parameter init for faster speed
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from models import VQVAE, build_vae_var

def main():
    # Model configuration
    print("hi1")
    MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====
    assert MODEL_DEPTH in {16, 20, 24, 30}

    # Download checkpoint
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt_file, var_ckpt_file = 'fsq-n_embed_64k.pt', 'ar-ckpt-best.pth'
    print("hi2")
    
    print("Downloading checkpoints...")
    if not osp.exists(vae_ckpt_file): 
        os.system(f'wget {hf_home}/{vae_ckpt_file}')
    if not osp.exists(var_ckpt_file): 
        os.system(f'wget {hf_home}/{var_ckpt_file}')

    # Build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Building models...")
    fsq_levels = [8, 8, 8, 5, 5, 5]  # Changed: 6 levels, product 64000
    vae = VQVAE(
        channel=512,
        z_channels=len(fsq_levels), # This will now be 6
        levels=fsq_levels,
    )
    print("hi3")

    # Load VAR checkpoint first to get the correct configuration
    var_checkpoint = torch.load(var_ckpt_file, map_location='cpu')
    checkpoint_args = var_checkpoint['args']
    
    # Use checkpoint configuration for proper model initialization
    checkpoint_patch_nums = checkpoint_args['patch_nums']
    checkpoint_depth = checkpoint_args['depth']
    
    print(f"Using checkpoint configuration:")
    print(f"  patch_nums: {checkpoint_patch_nums}")
    print(f"  depth: {checkpoint_depth}")
    
    var = VAR(
        vae_local=vae,
        patch_nums=checkpoint_patch_nums,
        num_classes=1000,  # Changed: Match checkpoint (ImageNet 1000 classes)
        depth=checkpoint_depth,
    )

    # Load checkpoints
    print("Loading checkpoints...")
    vae.load_state_dict(torch.load(vae_ckpt_file, map_location='cpu'), strict=True)
    
    # Load VAR checkpoint - extract model state dict from trainer checkpoint
    var_checkpoint = torch.load(var_ckpt_file, map_location='cpu')
    
    # Get configuration from checkpoint
    checkpoint_args = var_checkpoint['args']
    print(f"Checkpoint patch_nums: {checkpoint_args['patch_nums']}")
    print(f"Checkpoint depth: {checkpoint_args['depth']}")
    
    var_state_dict = var_checkpoint['trainer']['var_wo_ddp']
    
    # Load with non-strict mode to handle mismatched parameters
    missing_keys, unexpected_keys = var.load_state_dict(var_state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    vae.eval()
    var.eval()
    print("hi4")

    import pdb; pdb.set_trace()
    
    # Move models to device
    vae.to(device)
    var.to(device)
    
    # Disable gradients
    for p in vae.parameters(): 
        p.requires_grad_(False)
    for p in var.parameters(): 
        p.requires_grad_(False)
    
    print('Model preparation finished.')

    ############################# 2. Sample with classifier-free guidance

    # Set args
    seed = 0
    num_sampling_steps = 250
    cfg = 4
    class_labels = (980, 980, 437, 437, 22, 22, 562, 562)
    more_smooth = False  # True for more smooth output

    print(f"Sampling with seed: {seed}, cfg: {cfg}, class_labels: {class_labels}")

    # Set seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enable faster computation
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # Sample
    B = len(class_labels)
    label_B = torch.tensor(class_labels, device=device)
    
    print("Starting sampling...")
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            recon_B3HW = var.autoregressive_infer_cfg(
                B=B, 
                label_B=label_B, 
                cfg=cfg, 
                top_k=900, 
                top_p=0.95, 
                g_seed=seed, 
                more_smooth=more_smooth
            )

    # Create and save image grid

    import pdb; pdb.set_trace()
    print("Creating image grid...")
    chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
    chw = PImage.fromarray(chw.astype(np.uint8))
    
    # Save the image instead of showing it (since we're not in a GUI environment)
    output_path = 'generated_samples.png'
    chw.save(output_path)
    print(f"Generated samples saved to: {output_path}")

if __name__ == "__main__":
    main()
