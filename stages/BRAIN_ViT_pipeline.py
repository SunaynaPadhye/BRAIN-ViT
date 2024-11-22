import torch
import torch.nn as nn
import os
import sys

# Add the project root to sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.patchify_la import Patchify3D
from models.patch_selector_la import PatchSelector  # Import the new PatchSelector
from models.encoder_la import EncoderBlock
from models.classifier import ClassificationHead
from config.BRAIN_ViT_config import get_config
config = get_config()

class DownsampleBlock(nn.Module):
    """
    Downsamples spatial dimensions and doubles the number of channels.
    """
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)  # Halves D, H, W and doubles channels

class Stage(nn.Module):
    """
    A stage with multiple encoder blocks and patch selection.
    """
    def __init__(self, num_encoders, in_channels, out_channels, k, num_heads, viz_attn_weights=False, viz_topk=False, stage=1):
        super(Stage, self).__init__()
        self.patch_selector = PatchSelector(in_channels, out_channels, k, stage)  # Use new PatchSelector
        self.encoders = nn.ModuleList([
            EncoderBlock(channels=out_channels, num_heads=num_heads, viz_attn_weights=viz_attn_weights) for _ in range(num_encoders)
        ])
        self.viz_attn_weights = viz_attn_weights
        self.viz_topk = viz_topk

    def forward(self, x, M, patch_shape):
        attn_weight = []
        # Use PatchSelector to extract top patches
        top_patches, top_k_indices, _ = self.patch_selector(x)

        # Flatten the output to [Batch, Total_Patches, Channels]
        batch_size, channels, d, h, w = x.size()
        total_patches = d * h * w  # Total number of patches per sample
        x = x.permute(0, 2, 3, 4, 1).reshape(batch_size, total_patches, channels)
        M = M.permute(0, 2, 3, 4, 1).reshape(batch_size, total_patches, 1)

        for encoder in self.encoders:
            x, weights = encoder(x, top_patches, M, top_k_indices) # Query from x and KV from top patches
            attn_weight.append(weights)

        # Reshape the tensor back to [B, C, D, H, W]
        B, _, C = x.shape  # [B, Total_Patches, Channels]
        d, h, w = patch_shape
        x = x.reshape(B, C, d, h, w)

        return x, top_k_indices, attn_weight

class MultiStagePipeline(nn.Module):
    """
    Implements the complete multi-stage pipeline with patchification, downsampling, and classification.
    """
    def __init__(self, config=config):
        super(MultiStagePipeline, self).__init__()

        self.viz_attn_weights = config['viz_attn_weights']
        self.viz_topk = config['viz_topk']

        self.patchify = Patchify3D(config, in_channels=1)  # Patchification layer

        self.stage1 = Stage(config["N1"], config["patch_channels"], config["channels_stage1"], config["k1"], config["num_heads"], self.viz_attn_weights, self.viz_topk, stage=1)
        self.stage2 = Stage(config["N2"], config["channels_stage2"], config["channels_stage2"], config["k2"], config["num_heads"], self.viz_attn_weights, self.viz_topk, stage=2)
        self.stage3 = Stage(config["N3"], config["channels_stage3"], config["channels_stage3"], config["k3"], config["num_heads"], self.viz_attn_weights, self.viz_topk, stage=3)

        self.down1 = DownsampleBlock(config["channels_stage1"], config["channels_stage2"])
        self.down2 = DownsampleBlock(config["channels_stage2"], config["channels_stage3"])

        #for atlas scores
        self.max_pool_1 = nn.MaxPool3d(kernel_size=6, stride=6)
        self.max_pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.max_pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.classifier = ClassificationHead(in_channels=config["channels_stage3"])

        self.config = config

    def cosine_positional_embedding(self, x):
        """Generate cosine positional embeddings for 3D input [B, C, D, H, W]."""
        B, C, D, H, W = x.size()

        # Generate range tensors for each dimension
        d_range = torch.arange(D, dtype=torch.float32, device=x.device).unsqueeze(1)  # [D, 1]
        h_range = torch.arange(H, dtype=torch.float32, device=x.device).unsqueeze(1)  # [H, 1]
        w_range = torch.arange(W, dtype=torch.float32, device=x.device).unsqueeze(1)  # [W, 1]

        # Compute cosine embeddings for each axis
        depth_embedding = self.cosine_embedding(d_range, C).unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [D, 1, 1, 1, C]
        height_embedding = self.cosine_embedding(h_range, C).unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, H, 1, 1, C]
        width_embedding = self.cosine_embedding(w_range, C).unsqueeze(0).unsqueeze(1).unsqueeze(3)  # [1, 1, W, 1, C]

        # Expand embeddings to match the output shape for broadcasting
        depth_embedding = depth_embedding.expand(D, H, W, 1, C)  # [D, H, W, 1, C]
        height_embedding = height_embedding.expand(D, H, W, 1, C)  # [D, H, W, 1, C]
        width_embedding = width_embedding.expand(D, H, W, 1, C)  # [D, H, W, 1, C]

        # Sum the embeddings along the last dimension and permute to match [B, C, D, H, W]
        pos_embedding = (depth_embedding + height_embedding + width_embedding).squeeze(-2)  # [D, H, W, C]
        return pos_embedding.permute(3, 0, 1, 2).unsqueeze(0).to(x.device)  # [1, C, D, H, W]

    def cosine_embedding(self, pos, dim):
        """Compute cosine positional embeddings."""
        omega = torch.arange(dim, dtype=torch.float32, device=pos.device) / dim
        omega = 1.0 / (10000 ** (2 * (omega // 2) / dim))
        return torch.cos(pos * omega)  # [Pos, Dim]



    def forward(self, x, config=config):
        self.viz_attn_weights = config['viz_attn_weights']
        self.viz_topk = config['viz_topk']

        all_attn = dict()
        all_topk = dict()

        # Patchify input
        x, M = self.patchify(x)  # [B, patch_channels, D, H, W]
        pos_embedding = self.cosine_positional_embedding(x)
        x = x + pos_embedding

        M1 = self.max_pool_1(M.float())
        M2 = self.max_pool_2(M1)
        M3 = self.max_pool_3(M2)

        # Extract patch shape for later reshaping
        _, _, d, h, w = x.shape

        # Stage 1
        x, top_k1_indices, attn_weights = self.stage1(x, M1, (d, h, w)) 
        if self.viz_attn_weights:
            all_attn['stage1'] = attn_weights
        if self.viz_topk:
            all_topk['stage1'] = top_k1_indices

        x = self.down1(x)  # Downsample

        # Stage 2
        _, _, d, h, w = x.shape
        x, top_k2_indices, attn_weights = self.stage2(x, M2, (d, h, w)) 
        if self.viz_attn_weights:
            all_attn['stage2'] = attn_weights
        if self.viz_topk:
            all_topk['stage2'] = top_k2_indices

        x = self.down2(x)  # Downsample

        # Stage 3
        _, _, d, h, w = x.shape
        x, top_k3_indices, attn_weights = self.stage3(x, M3, (d, h, w)) 
        if self.viz_attn_weights:
            all_attn['stage3'] = attn_weights
        if self.viz_topk:
            all_topk['stage3'] = top_k3_indices

        # Classification head
        logits = self.classifier(x)

        if self.viz_attn_weights or self.viz_topk:
            return logits, all_attn, all_topk
        else:
            return logits
