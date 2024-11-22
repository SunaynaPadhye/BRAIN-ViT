import torch
import torch.nn as nn
import os
import sys
import torch.nn.functional as F
# Add the project root to sys.path to ensure config imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


class EncoderBlock(nn.Module):
    """
    A single encoder block with LayerNorm, MHA, MLP, and residual connections.
    """
    def __init__(self, channels, num_heads=4, initial_alpha=0.5, viz_attn_weights=False):
        super(EncoderBlock, self).__init__()

        # LayerNorms for X and Top-K patches (I)
        self.norm1_x = nn.LayerNorm(channels)  
        self.norm1_i = nn.LayerNorm(channels)  

        # Linear layers to compute Q, K, V
        self.q_linear = nn.Linear(channels, channels)
        self.k_linear = nn.Linear(channels, channels)
        self.v_linear = nn.Linear(channels, channels)
        self.num_heads = num_heads

        # Multi-Head Attention
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

        # LayerNorm and MLP for final residual block
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, 2*channels),
            nn.GELU(),
            nn.Linear(2*channels, channels)
        )

        # Learnable parameters
        self.w2 = nn.Parameter(torch.tensor(1.0))  # Initialize w2
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))  # Initialize alpha as learnable, centered around zero

        self.viz_attn_weights = viz_attn_weights

    def get_bias(self, M, top_k_indices):
        alpha_scaled = (torch.tanh(self.alpha + 1e-7) + 1) / 2
        w2 = F.softplus(self.w2 + 1e-7)
        w1 = alpha_scaled * w2

        # Separate the scores of 1 and 2
        mask_1 = (M == 1.).float()
        mask_2 = (M == 2.).float()
         
        # Apply weights to the scores of 1 and 2
        M_weighted = w1 * mask_1 + w2 * mask_2

        top_k_scores = torch.gather(M_weighted.squeeze(-1), 1, top_k_indices)
        attn_bias = torch.bmm(M_weighted, top_k_scores.unsqueeze(1))
        normalized_learned_scores = F.normalize(attn_bias, p=2, dim=-1)
        # Prepare attn_mask for nn.MultiheadAttention
        attn_bias = normalized_learned_scores.repeat_interleave(self.num_heads, dim=0)  # Shape: (batch_size * num_heads, num_query, num_k)

        return attn_bias

    def forward(self, x, i, M, top_k_indices):
        """
        Forward pass through the encoder block.
        Input:
            x: [Batch, Total_Patches, Channels] - Output from previous layer/downsampling.
            i: [Batch, Top_K, Channels] - Top-K patches selected.
        Output:
            x: [Batch, Total_Patches, Channels] - Processed output.
        """

        # Step 1: Apply LayerNorm
        x_norm = self.norm1_x(x)  # [B, Total_Patches, Channels]
        i_norm = self.norm1_i(i)  # [B, Top_K, Channels]

        # Step 2: Compute Q, K, V using learnable linear layers
        Q = self.q_linear(x_norm)  # [B, Total_Patches, Channels]
        K = self.k_linear(i_norm)  # [B, Top_K, Channels]
        V = self.v_linear(i_norm)  # [B, Top_K, Channels]

        # Step 3: Apply MHA
        attn_bias = self.get_bias(M, top_k_indices)
        attn_output, attn_weights = self.mha(Q, K, V, attn_mask=attn_bias)  # [B, Total_Patches, Channels]

        # Step 4: Residual Add and LayerNorm
        x = x + attn_output  # Residual Add
        x_norm = self.norm2(x)  # Apply LayerNorm

        # Step 5: Pass through MLP and Residual Add
        mlp_output = self.mlp(x_norm)  # [B, Total_Patches, Channels]
        x = x + mlp_output  # Residual Add

        # if self.viz_attn_weights:
        return x, attn_weights  # Output for the next encoder or stage
     