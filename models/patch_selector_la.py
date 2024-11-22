import torch
import torch.nn as nn
import os
import sys

# Ensure project root is in sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import CNNFeatureExtractor from cnn_blocks.py
from models.dwc_pwc_blocks import CNNFeatureExtractor

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.BRAIN_ViT_config import get_config
config = get_config()
lookup_dir = config["lookup_dir"]
alloc_dir = config["alloc_dir"]

class ProbeMLP(nn.Module):
    """
    MLP to compute importance scores and select top-K patches.
    Now with input, hidden, and output layers.
    """
    def __init__(self, input_dim, stage, temperature=1.0, lookup_dir=lookup_dir, alloc_dir=alloc_dir):
        super(ProbeMLP, self).__init__()
        
        # Define MLP layers
        hidden_dim = input_dim // 2  # Hidden layer with half the input dimension
        
        self.input_norm = nn.LayerNorm(input_dim)  # Normalize input
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input to hidden layer
        self.relu = nn.ReLU()  # Non-linearity to prevent vanishing gradient
        self.fc2 = nn.Linear(hidden_dim, 1)  # Hidden to output layer
        
        # self.k = k  # Number of top-K patches to select
        self.temperature = temperature  # Temperature scaling for Gumbel-Softmax
        self.stage = stage
        
        # Load lookup and allocation for the specified stage
        self.lookup = nn.Parameter(torch.load(os.path.join(lookup_dir, f"stage{stage}_lookup.pt"), weights_only=False), requires_grad=False)
        self.allocations = torch.load(os.path.join(alloc_dir, f"stage{stage}_allocations.pt"), weights_only=False)
        self.labels, self.values = self.get_allocations(self.allocations)

    def get_allocations(self, allocations):
        label = allocations.keys()
        m = allocations.values()
        return nn.Parameter(torch.tensor(list(label)), requires_grad=False), nn.Parameter(torch.tensor(list(m)), requires_grad=False)

    def forward(self, patches, features):
        # Enable gradient tracking on features
        features = features.requires_grad_()

        # Pass through input layer -> ReLU -> output layer
        hidden_output = self.fc1(features)  # [B, Total_Patches, Hidden_Dim]
        hidden_output = self.relu(hidden_output)  # Non-linearity
        importance_logits = self.fc2(hidden_output).squeeze(-1)  # [B, Total_Patches]

        # Apply Gumbel-Softmax for differentiable top-K selection
        gumbel_softmax_scores = F.gumbel_softmax(importance_logits, tau=self.temperature, hard=False, dim=-1)

        batch_size = patches.size(0)
        total_patches = patches.size(1)  # This should be h*w*d

        # Reshape lookup to match patches shape and replicate for batch
        lookup_flat = self.lookup.reshape(-1)[:total_patches]  # Flatten 3D to 1D
        lookup_flat = lookup_flat.unsqueeze(0).expand(batch_size, -1)  # Add batch dimension
        # lookup_flat = lookup_flat.to(gumbel_softmax_scores.device)

        # Initialize tensor to store selected indices for each batch
        all_selected_indices = torch.zeros((batch_size, sum(self.values)), 
                                        dtype=torch.long, 
                                        device=gumbel_softmax_scores.device)
        
        # Keep track of position in the indices tensor
        current_position = 0

        # Loop through each parcel
        for i in range(self.labels.shape[0]):
            # Create mask for current parcel (for all batches)
            parcel_mask = (lookup_flat == self.labels[i])  # Shape: [batch_size, total_patches]
            
            if parcel_mask.any():
                # For each batch
                for batch_idx in range(batch_size):
                    # Get scores for current parcel
                    parcel_scores = gumbel_softmax_scores[batch_idx].clone()
                    
                    # Mask out non-parcel indices
                    parcel_scores[~parcel_mask[batch_idx]] = float('-inf')
                    # print(parcel_scores.device)

                    # Select top-m indices
                    if self.values[i] > 0:
                        # Get number of available patches for this parcel
                        available_patches = parcel_mask[batch_idx].sum()
                        k = min(self.values[i], available_patches)
                        
                        if k > 0:
                            _, top_m = torch.topk(parcel_scores, k)
                            # Store indices at the correct position
                            all_selected_indices[batch_idx, current_position:current_position + k] = top_m
                
                current_position += self.values[i]

        # Create final mask and apply gradients
        final_mask = torch.zeros_like(gumbel_softmax_scores)
        for batch_idx in range(batch_size):
            final_mask[batch_idx].scatter_(0, all_selected_indices[batch_idx], 1.0)

        # Masks using Gumbel-Softmax scores
        mask_grad = final_mask + importance_logits.detach() - importance_logits
        final_patches = patches * mask_grad.unsqueeze(-1)

        # Gather selected patches using torch.gather
        selected_patches = torch.gather(
            final_patches, 1,
            all_selected_indices.unsqueeze(-1).expand(-1, -1, patches.size(-1))
        )

        return selected_patches, all_selected_indices, importance_logits

class PatchSelector(nn.Module):
    """
    Combines CNNFeatureExtractor with ProbeMLP to extract and select top-K patches.
    """
    def __init__(self, in_channels, out_channels, k, stage):
        super(PatchSelector, self).__init__()
        self.feature_extractor = CNNFeatureExtractor(in_channels, out_channels)
        self.probe_mlp = ProbeMLP(input_dim=out_channels, stage=stage)

    def forward(self, patches):
        # Extract features using CNN
        features = self.feature_extractor(patches)

        # Reshape patches and features to [B, Total_Patches, C]
        batch_size, channels, d, h, w = patches.size()
        total_patches = d * h * w
        patches = patches.permute(0, 2, 3, 4, 1).reshape(batch_size, total_patches, channels)

        # Select top-K patches using ProbeMLP
        selected_patches, selected_indices, scores_softmax = self.probe_mlp(patches, features)

        return selected_patches, selected_indices, scores_softmax

def main():
    # Configuration
    batch_size = 2
    patch_channels = 64  # Number of channels from patchification
    d_init, h_init, w_init = 31, 37, 31  # Dimensions of input patches
    k = 10  # Number of top-K patches to select

    # Create a dummy input tensor: [Batch, Channels, D, H, W]
    dummy_input = torch.randn(batch_size, patch_channels, d_init, h_init, w_init)

    # Initialize PatchSelector
    patch_selector = PatchSelector(
        in_channels=patch_channels, 
        out_channels=patch_channels, 
        k=k
    )

    # Forward pass through the patch selector
    top_patches, top_k_indices, scores_softmax = patch_selector(dummy_input)

    # Print results
    print(f"Top Patches Shape: {top_patches.shape}")  # [B, K, Channels]
    print(f"Top-K Indices: {top_k_indices}")
    print(f"Scores Softmax: {scores_softmax}")

if __name__ == "__main__":
    main()
