# CNN feature extraction module.
# Contains convolution blocks with BatchNorm and ReLU activation.


import torch
import torch.nn as nn
import os
import sys

# Add project root to sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A single 3D convolutional block with Depthwise Conv, Pointwise Conv, BatchNorm, and GELU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # Depthwise convolution
        self.depthwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.bn1 = nn.BatchNorm3d(in_channels, eps=1e-3)
        self.bn2 = nn.BatchNorm3d(in_channels, eps=1e-3)
        self.bn3 = nn.BatchNorm3d(out_channels, eps=1e-3)
        # Pointwise convolution
        self.pointwise_conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.pointwise_conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        # GELU Activation
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.gelu3 = nn.GELU()

    def forward(self, x):
        # Depthwise Convolution
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.gelu1(x)
        # Pointwise Convolution
        x = self.pointwise_conv1(x)
        x = self.bn2(x)
        x = self.gelu2(x)
        x = self.pointwise_conv2(x)
        x = self.bn3(x)
        x = self.gelu3(x)
        return x

class CNNFeatureExtractor(nn.Module):
    """
    Three stacked 3D convolutional blocks for feature extraction.
    """
    def __init__(self, in_channels, out_channels):
        super(CNNFeatureExtractor, self).__init__()
        self.layer1 = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        # Forward through each layer
        x = self.layer1(x)  # [Batch, Channels, D, H, W]

        # Flatten the output to [Batch, Total_Patches, Channels]
        batch_size, channels, d, h, w = x.size()
        total_patches = d * h * w  # Total number of patches per sample

        # Reshape to [Batch, Total_Patches, Channels]
        x = x.permute(0, 2, 3, 4, 1).reshape(batch_size, total_patches, channels)
        return x
