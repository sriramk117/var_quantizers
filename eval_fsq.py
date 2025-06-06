import argparse
import torch
import os
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import random

# Assuming VAR model is in models.basic_var and FSQ is used internally
from models.basic_var import VAR 
# FSQ class itself, though VAR handles its instantiation
from utils.fsq import FSQ 
from utils.misc import create_npz_from_sample_folder

# Simplified seed setup for single-process script, inspired by utils.misc.setup_dist_and_seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine model parameters
    # As per train.py logic: patch_num_for_model = img_size // patch_size (default patch_size=16)
    patch_size_val = 16 # VAR model's default patch_size
    patch_num_for_model = args.image_size // patch_size_val

    fsq_levels_list = [int(l.strip()) for l in args.fsq_levels.split(',')]
    vq_config = {'levels': fsq_levels_list}
    
    print(f"Initializing VAR model with: image_size={args.image_size}, depth={args.depth}, patch_num={patch_num_for_model}, FSQ levels={fsq_levels_list}")

    model = VAR(
        img_size=args.image_size,
        patch_size=patch_size_val, # Explicitly pass patch_size
        patch_num=patch_num_for_model,
        depth=args.depth,
        vq_type='fsq',
        vq_cfg=vq_config,
        # Assuming other VAR parameters (hidden_size, num_heads, etc.) match the checkpoint's training defaults
        # or are not crucial to change for FSQ evaluation if checkpoint handles them.
    ).to(device)

    print(f"Loading checkpoint from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    # Remove 'module.' prefix if present (from DDP training)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state_dict)
    model.eval()
    print("Model loaded successfully.")

    # Prepare for sampling
    os.makedirs(args.output_dir, exist_ok=True)
    samples_png_folder = os.path.join(args.output_dir, "samples_png")
    os.makedirs(samples_png_folder, exist_ok=True)

    total_samples_generated = 0
    # As per README: "sample 50,000 images (50 per class)" implies 1000 classes for ImageNet
    samples_per_class = args.num_samples // args.num_classes
    if args.num_samples % args.num_classes != 0:
        print(f"Warning: num_samples ({args.num_samples}) is not perfectly divisible by num_classes ({args.num_classes}). Adjusting samples per class.")

    print(f"Starting sampling: target {args.num_samples} images, approx. {samples_per_class} per class for {args.num_classes} classes.")

    for class_idx in tqdm(range(args.num_classes), desc="Sampling per class"):
        if total_samples_generated >= args.num_samples:
            break

        num_to_generate_for_this_class = min(samples_per_class, args.num_samples - total_samples_generated)
        if num_to_generate_for_this_class <= 0:
            continue
            
        cls_labels = torch.tensor([class_idx] * num_to_generate_for_this_class, device=device, dtype=torch.long)
        
        with torch.no_grad():
            # VAR.autoregressive_infer_cfg(self, bs, cls_label, cfg=1.5, top_k=None, top_p=None, more_smooth=False, progress=False)
            sampled_images = model.autoregressive_infer_cfg(
                bs=num_to_generate_for_this_class,
                cls_label=cls_labels,
                cfg=args.cfg_scale,
                top_k=args.top_k if args.top_k > 0 else None,
                top_p=args.top_p if args.top_p > 0 else None,
                more_smooth=args.more_smooth,
                progress=False # tqdm provides outer loop progress
            )
        
        # Normalize images from [-1, 1] (typical VAE/GAN output) to [0, 1] for save_image
        sampled_images = (sampled_images + 1) / 2.0
        sampled_images = sampled_images.clamp(0, 1)

        for i in range(sampled_images.size(0)):
            if total_samples_generated >= args.num_samples:
                break
            img_filename = os.path.join(samples_png_folder, f"class_{class_idx:04d}_sample_{total_samples_generated:05d}.png")
            save_image(sampled_images[i], img_filename)
            total_samples_generated += 1
            
    print(f"Generated {total_samples_generated} images in {samples_png_folder}")

    # Create NPZ file
    npz_filename = f"imagenet_{args.image_size}x{args.image_size}_fsq_samples_{total_samples_generated}.npz"
    npz_path = os.path.join(args.output_dir, npz_filename)
    print(f"Creating NPZ file at {npz_path}...")
    
    create_npz_from_sample_folder(
        folder=samples_png_folder, 
        npz_path=npz_path, 
        resize_size=(args.image_size, args.image_size),
        sort_key_func=None # Default sort should be fine with current naming
    )
    print(f"NPZ file created: {npz_path}")
    print("\nEvaluation samples prepared.")
    print("To calculate FID, IS, Precision, Recall, use OpenAI's FID evaluation toolkit.")
    print(f"Example command for FID:")
    print(f"python -m pytorch_fid \"{npz_path}\" \"/path/to/imagenet_val_stats_{args.image_size}.npz\" --device {device}")
    print(f"Replace '/path/to/imagenet_val_stats_{args.image_size}.npz' with the actual path to your pre-calculated ImageNet validation set statistics NPZ file for the corresponding image size.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for VAR models using FSQ.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint (.pth file).")
    parser.add_argument("--output_dir", type=str, default="fsq_eval_output", help="Directory to save generated samples and NPZ file.")
    
    # Model architecture parameters (must match the loaded checkpoint)
    parser.add_argument("--depth", type=int, required=True, help="Depth of the VAR model (e.g., 16, 24, 30).")
    parser.add_argument("--image_size", type=int, default=256, choices=[256, 512], help="Image size (e.g., 256, 512). This corresponds to --pn in training scripts.")
    parser.add_argument("--fsq_levels", type=str, required=True, help="Comma-separated list of FSQ levels (e.g., '8,6,5,5'). Must match the FSQ configuration of the loaded model.")

    # Sampling parameters
    parser.add_argument("--num_samples", type=int, default=50000, help="Total number of samples to generate.")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of ImageNet classes for class-conditional sampling.")
    parser.add_argument("--cfg_scale", type=float, default=1.5, help="Classifier-free guidance scale.")
    parser.add_argument("--top_k", type=int, default=900, help="Top-k sampling parameter. Set to 0 to disable.")
    parser.add_argument("--top_p", type=float, default=0.96, help="Top-p (nucleus) sampling parameter. Set to 0.0 to disable.")
    parser.add_argument("--more_smooth", action="store_true", help="Use 'more_smooth' option in sampling for potentially better visual quality (as per README).")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()

    # Validate FSQ levels format
    try:
        levels = [int(l.strip()) for l in args.fsq_levels.split(',')]
        if not levels or not all(isinstance(l, int) and l > 0 for l in levels):
            raise ValueError("FSQ levels must be positive integers.")
    except ValueError as e:
        parser.error(f"Invalid --fsq_levels: {e}. Must be comma-separated positive integers (e.g., '8,6,5').")

    main(args)