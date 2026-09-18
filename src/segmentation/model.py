"""
UNet2DModel wrapper for binary burn-scar segmentation.

UNet2DModel (from diffusers) was designed for denoising diffusion. Here we
repurpose it as a plain encoder-decoder by always passing timestep=0, which
makes the sinusoidal timestep embedding near-zero and effectively inert.
"""

import torch
import torch.nn as nn
from diffusers import UNet2DModel


class BurnScarSegmenter(nn.Module):
    """
    6-channel → 1-channel binary segmentation UNet.

    Architecture constants are class-level so checkpoints can be reproduced
    exactly without needing to pass constructor arguments.
    """

    BLOCK_OUT_CHANNELS = (64, 128, 256, 512)
    LAYERS_PER_BLOCK   = 2
    DOWN_BLOCK_TYPES   = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D")
    UP_BLOCK_TYPES     = ("UpBlock2D",   "UpBlock2D",   "UpBlock2D",   "UpBlock2D")

    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet2DModel(
            in_channels=6,
            out_channels=1,
            block_out_channels=self.BLOCK_OUT_CHANNELS,
            layers_per_block=self.LAYERS_PER_BLOCK,
            down_block_types=self.DOWN_BLOCK_TYPES,
            up_block_types=self.UP_BLOCK_TYPES,
            norm_num_groups=32,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) normalised HLS image tensor
        Returns:
            (B, 1, H, W) raw logits (pre-sigmoid)
        """
        timestep = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.unet(x, timestep=timestep).sample

    def save_pretrained(self, output_dir: str) -> None:
        """Save in HuggingFace diffusers format for easy reloading."""
        self.unet.save_pretrained(output_dir)

    @classmethod
    def load_pretrained(cls, checkpoint_dir: str) -> "BurnScarSegmenter":
        """Reload from a HuggingFace-format checkpoint directory."""
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.unet = UNet2DModel.from_pretrained(checkpoint_dir)
        return instance
