# Patchification using 3D convolution.
# Converts the sMRI input into non-overlapping patches.

import torch
import torch.nn as nn
import os
import sys
import math

# Add the project root to sys.path to ensure config imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.BRAIN_ViT_config import get_config
config = get_config()
h_init = config["init_shape"][0]
w_init = config["init_shape"][1]
d_init = config["init_shape"][2]
patch_kernel = config["patch_kernel"]
patch_stride = config["patch_stride"]
patch_channels = config["patch_channels"]

from data.atlas_scores.get_scores import process_batch, prepare_regions

class Patchify3D(nn.Module):
    """
    Converts input sMRI data into non-overlapping patches using Conv3D.
    Applies padding to ensure input dimensions are divisible by patch sizes.
    """
    def __init__(self, config, in_channels=1):
        super(Patchify3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, patch_channels,
            kernel_size=patch_kernel,
            stride=patch_stride
        )
        self.high_relevance_mask = prepare_regions()
        #for atlas scores
        self.max_pool = nn.MaxPool3d(kernel_size=patch_kernel, stride=patch_stride)

    def forward(self, x):
        atlas_data_scores = process_batch(x, self.high_relevance_mask.unsqueeze(0))
        # Input: [B, C, D, H, W] (Batch, Channels, Depth, Height, Width)

        #Pad the data equally on all sides to make it 184x220x184 (h w d)

        # Calculate padding required to make dimensions divisible by patch sizes
        pad_d = (patch_stride[2] - d_init % patch_stride[2]) % patch_stride[2]
        pad_h = (patch_stride[0] - h_init % patch_stride[0]) % patch_stride[0]
        pad_w = (patch_stride[1] - w_init % patch_stride[1]) % patch_stride[1]

        # Pad equally on both sides: (left, right, top, bottom, front, back)
        padding = (pad_w // 2, pad_w - pad_w // 2,
                   pad_h // 2, pad_h - pad_h // 2,
                   pad_d // 2, pad_d - pad_d // 2)

        # Apply padding
        x = nn.functional.pad(x, padding, mode='constant', value=0)
        atlas_data_scores = nn.functional.pad(atlas_data_scores, padding, mode='constant', value=0)

        # Forward pass through Conv3D to get patches
        patchified_output = self.conv(x)  # [B, C, D, H, W]
        
        return patchified_output, atlas_data_scores
