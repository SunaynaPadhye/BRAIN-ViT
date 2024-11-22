# Contains all hyperparameters, dimensions, and channel configurations.
# Manages parameters for batch size, patch sizes, number of encoders per stage, etc.

import torch
import os

# Initial dimensions of sMRI input
h_init, w_init, d_init = 182, 218, 182  # (Height, Width, Depth)
# total patches 116380
batch_size = 24

# Patchification parameters
patch_kernel = (6, 6, 6)  # Kernel size for patchification
patch_stride = (6, 6, 6)  # Stride for patchification
patch_channels = 64  # Number of channels after patchification

# Derived dimensions after patchification
height = h_init // patch_stride[0]  # 256 // 4 = 64
width = w_init // patch_stride[1]   # 256 // 4 = 64
depth = d_init // patch_stride[2]   # 168 // 4 = 42

# Channels for different stages
channels_stage1 = 64  # Channels within Stage 1
channels_stage2 = 128  # Channels within Stage 2
channels_stage3 = 256  # Channels within Stage 3
# channels_stage4 = 512  # Channels within Stage 4

k1 = 1500  # Number of top patches to select for attention for stage 1
k2 = 1000  # Number of top patches to select for attention for stage 2
k3 = 150  # Number of top patches to select for attention for stage 3

N1, N2, N3 = 3, 3, 3      # Ns = No. of encoders in each stage s : removed N4
num_heads = 4

# Classification settings
num_classes = 3  # CN - 0, AD - 1, MCI - 2


# Training Hyperparameters
num_epochs = 15          # Total number of epochs
learning_rate = 0.001    # Initial learning rate
weight_decay = 1e-5      # L2 regularization
T_max = 20
min_lr = 1e-5

# Paths for logs and checkpoints
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
logs_dir = os.path.join(project_root, "logs")
lookup_dir = os.path.join(project_root, "data/lookups_combined_modified")
alloc_dir = os.path.join(project_root, "data/patch_allocations_combined_modified")


# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
# device = "cpu"  # Use GPU if available

#attention visualization
viz_attn_weights = False
viz_topk = False


# Function to return all config values as a dictionary
def get_config():
    return {
        "batch_size": batch_size,
        "init_shape": (h_init, w_init, d_init),
        "patch_kernel": patch_kernel,
        "patch_stride": patch_stride,
        "patch_channels": patch_channels,
        "height": height,
        "width": width,
        "depth": depth,
        "channels_stage1": channels_stage1,
        "channels_stage2": channels_stage2,
        "channels_stage3": channels_stage3,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "N1": N1,
        "N2": N2,
        "N3": N3,
        "num_heads": num_heads,
        "num_classes": num_classes,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "T_max": T_max,
        "min_lr": min_lr,
        "logs_dir": logs_dir,
        "device": device,
        "viz_attn_weights": viz_attn_weights,
        "viz_topk": viz_topk,
        "lookup_dir":lookup_dir,
        "alloc_dir":alloc_dir

    }


