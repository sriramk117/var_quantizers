"""
This VQVAE model is adapted from the FSQ-PyTorch repository (duchenzhuang/FSQ-pytorch)
to match the architecture of the pretrained checkpoint. It replaces the original
U-Net-like VQVAE from the VAR repository.
"""
from typing import List, Any, Dict, Optional, Tuple, Sequence, Union

import torch
import torch.nn as nn

# Assuming fsq.py is in the same directory (models/)
from utils.fsq import FSQ

class Encoder(nn.Module):
    """
    A simple convolutional encoder that matches the architecture from
    the repository where the FSQ checkpoint was trained.
    """
    def __init__(self, in_channel, channel, embed_dim):
        super().__init__()
        blocks = [
            nn.Conv2d(in_channel, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, embed_dim, 1)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    """
    A simple convolutional decoder that matches the architecture from
    the repository where the FSQ checkpoint was trained.
    """
    def __init__(self, in_channel, out_channel, channel):
        super().__init__()
        blocks = [
            nn.ConvTranspose2d(in_channel, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, out_channel, 1)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    """
    The main VQ-VAE model class that integrates the Encoder, Decoder, and FSQ quantizer.
    Its structure is now compatible with the pretrained fsq-n_embed_1k.pt checkpoint.
    """
    def __init__(
        self,
        in_channel: int = 3,
        channel: int = 128,
        z_channels: int = 4, # This is the FSQ embedding dimension, must match levels
        levels: List[int] = [8, 5, 5, 5],
        **kwargs # Absorb unused arguments from train.py (like `dropout`) to prevent errors
    ):
        super().__init__()
        
        # In this architecture, `z_channels` is the embedding dimension for FSQ.
        # It must equal len(levels).
        if z_channels != len(levels):
            raise ValueError(f"z_channels ({z_channels}) must match the number of FSQ levels ({len(levels)})")
            
        self.Cvae = z_channels
        self.enc = Encoder(in_channel, channel, z_channels)
        
        # The FSQ quantizer from the original project was named 'quantize_t'.
        # We name it 'fsq' for clarity, and handle the key mapping in load_state_dict.
        self.fsq = FSQ(levels=levels, dim=z_channels)
        
        self.dec = Decoder(z_channels, in_channel, channel)
        
        # The total number of unique codes in the codebook.
        self.vocab_size = self.fsq.codebook_size

    def forward(self, input):
        """
        Defines the forward pass for VAE training/reconstruction.
        """
        h = self.enc(input)
        quant, indices = self.fsq(h)
        # The original FSQ repo returned a 'diff' term. FSQ doesn't have a commitment loss,
        # so we return a zero tensor for compatibility.
        diff = torch.tensor(0.0, device=input.device)
        dec = self.dec(quant)
        return dec, diff, indices

    @torch.no_grad()
    def img_to_idxBl(self, inp_img: torch.Tensor, patch_nums: Optional[Tuple[int, ...]] = None) -> List[torch.Tensor]:
        """
        FSQ equivalent of VAR's img_to_idxBl method.
        Creates multi-scale token sequences like the original VQ-VAE.
        
        Args:
            inp_img: Input image [B, 3, H, W]
            patch_nums: Patch numbers for each scale (e.g., (1, 2, 3, 4, 5, 6, 8, 10, 13, 16))
        
        Returns:
            List of token tensors [B, patch_num**2] for each scale
        """
        if patch_nums is None:
            patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  # Default VAR patch numbers
        
        h = self.enc(inp_img)  # [B, C, H, W]
        B, C, H, W = h.shape
        
        # This simulates the multi-scale quantization process that VQ-VAE does
        # We'll create a residual-like process for FSQ
        f_rest = h.clone()  # Start with the encoded features
        token_maps = []
        
        for i, patch_num in enumerate(patch_nums):
            # Downsample the remaining features to current patch size
            if patch_num != H:
                f_current = torch.nn.functional.interpolate(
                    f_rest, size=(patch_num, patch_num), mode='area'
                )
            else:
                f_current = f_rest
            
            # Quantize at current resolution
            quant_current, indices_current = self.fsq(f_current)
            
            # FSQ returns indices with shape [B, H, W] - already packed indices
            # Convert to long dtype for CrossEntropyLoss compatibility
            B_curr, H_curr, W_curr = indices_current.shape
            packed_indices = indices_current.long().reshape(B_curr, H_curr * W_curr)
            
            token_maps.append(packed_indices)
            
            # Update residual (subtract the upsampled quantized version)
            if i < len(patch_nums) - 1:  # Not the last scale
                if patch_num != H:
                    quant_upsampled = torch.nn.functional.interpolate(
                        quant_current, size=(H, W), mode='bicubic'
                    )
                else:
                    quant_upsampled = quant_current
                f_rest = f_rest - quant_upsampled
        
        return token_maps

    @torch.no_grad()
    def fsq_gt_idx_Bl(self, inp_img: torch.Tensor) -> List[torch.Tensor]:
        """
        Wrapper for img_to_idxBl to maintain backward compatibility.
        Returns the same multi-scale token maps but with the original method name.
        """
        return self.img_to_idxBl(inp_img)

    @torch.no_grad()
    def img_to_indices(self, inp_img: torch.Tensor) -> torch.Tensor:
        """
        Encodes an image to a 2D tensor of FSQ indices.
        """
        h = self.enc(inp_img)
        _quant, indices = self.fsq(h)
        return indices

    @torch.no_grad()
    def indices_to_img(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decodes a 2D tensor of FSQ indices back into an image.
        """
        # ---- START DEBUG ----
        print(f"[VQVAE.indices_to_img] self.fsq.dim: {self.fsq.dim}")
        print(f"[VQVAE.indices_to_img] self.fsq.num_codebooks: {self.fsq.num_codebooks}")
        print(f"[VQVAE.indices_to_img] self.fsq.project_out: {self.fsq.project_out}")
        print(f"[VQVAE.indices_to_img] input indices shape: {indices.shape}")
        # ---- END DEBUG ----
        quant = self.fsq.indices_to_codes(indices, project_out=True)
        # ---- START DEBUG ----
        print(f"[VQVAE.indices_to_img] quant shape after fsq.indices_to_codes: {quant.shape}")
        # ---- END DEBUG ----
        
        # Reshape quant from [B, L, C] to [B, C, H, W]
        # Assuming L = H * W, and H = W (square feature map)
        B, L, C = quant.shape
        H = W = int(L**0.5) # Calculate H and W, assuming square
        if H * W != L:
            raise ValueError(f"Cannot reshape [B, L, C] to [B, C, H, W] when L ({L}) is not a perfect square.")
        quant = quant.permute(0, 2, 1).reshape(B, C, H, W)
        # ---- START DEBUG ----
        print(f"[VQVAE.indices_to_img] quant shape after reshape: {quant.shape}")
        # ---- END DEBUG ----

        dec = self.dec(quant)
        return dec.clamp(-1, 1)

    @torch.no_grad()
    def idxBl_to_var_input(self, gt_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """
        Convert token indices back to continuous features for VAR input.
        This is used for teacher forcing in VAR training.
        
        Args:
            gt_idx_Bl: List of token tensors [B, patch_num**2] for each scale
        
        Returns:
            Continuous features [B, L, C] for VAR input
        """
        # For FSQ, the indices are already packed, so we need to convert them back 
        # to the spatial format that FSQ expects for indices_to_codes
        B = gt_idx_Bl[0].shape[0]
        device = gt_idx_Bl[0].device
        
        # Use the finest scale (last scale) which has the most detail
        packed_indices = gt_idx_Bl[-1]  # Use the finest scale
        H = W = int(packed_indices.shape[1] ** 0.5)  # Assume square
        
        # Reshape packed indices back to spatial format [B, H, W]
        indices_spatial = packed_indices.reshape(B, H, W)
        
        # Convert indices to continuous features using FSQ
        quant = self.fsq.indices_to_codes(indices_spatial, project_out=True)
        
        # Reshape to [B, L, C] format expected by VAR
        B, C, H, W = quant.shape
        quant_BLC = quant.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        return quant_BLC

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """
        Custom load_state_dict to handle key mapping from the pretrained checkpoint.
        This function translates the keys from the saved model (e.g., 'module.quantize_t.some_weight')
        to match the keys in this model (e.g., 'fsq.some_weight').
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if it was saved from a DDP model
            if k.startswith('module.'):
                k = k[len('module.'):]
            
            # Remap the quantizer keys from 'quantize_t' to 'fsq'
            if k.startswith('quantize_t.'):
                k = k.replace('quantize_t.', 'fsq.', 1)
            
            new_state_dict[k] = v
            
        # We use strict=False because the FSQ checkpoint was trained with an older
        # version of the FSQ class and may not have the buffers `_levels` and `_basis`.
        # PyTorch will correctly initialize them from the constructor.
        return super().load_state_dict(new_state_dict, strict=False)


    # def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
    #     """Loads a pretrained FSQ VQVAE model and extracts the weights."""
    #     full_ckpt = torch.load(ckpt_path, map_location='cpu')
        
    #     # This function assumes the checkpoint comes from a model like the one in `train_fsq.py`
    #     # and may need key adjustments.
    #     if 'model_state_dict' in full_ckpt:
    #         full_ckpt = full_ckpt['model_state_dict']

    #     model_state_dict = self.state_dict()
    #     pretrained_dict = {k.replace('module.', ''): v for k, v in full_ckpt.items()}
        
    #     # Filter out unnecessary keys and update the current model's state dict
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
        
    #     model_state_dict.update(pretrained_dict)
    #     if 'fsq.implicit_codebook' in model_state_dict and model_state_dict['fsq.implicit_codebook'].shape[0] != self.fsq.implicit_codebook.shape[0]:
    #         model_state_dict['fsq.implicit_codebook'] = self.fsq.implicit_codebook
    #     return super().load_state_dict(state_dict=model_state_dict, strict=strict, assign=assign)
    #     # print(f"Loaded {len(pretrained_dict)} matching keys from pretrained FSQ VQ-VAE: {ckpt_path}")