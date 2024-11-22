import torch
import torch.nn as nn
from config.BRAIN_ViT_config import get_config
config = get_config()

class ClassificationHead(nn.Module):
    """
    Classification head with Global Average Pooling followed by a Linear layer.
    """
    def __init__(self, in_channels):
        super(ClassificationHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # Pool to 1x1x1
        self.fc1 = nn.Linear(in_channels, 128)  # Linear layer for classification
        self.fc2 = nn.Linear(128, 32)  # Linear layer for classification
        self.fc3 = nn.Linear(32, config["num_classes"])  # Linear layer for classification

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        

    def forward(self, x):
        """
        Input: [Batch, Channels, Depth, Height, Width]
        Output: [Batch, num_classes] (Logits)
        """
        # Global Average Pooling to get [Batch, Channels, 1, 1, 1]
        x = self.global_avg_pool(x).view(x.size(0), -1)  # Reshape to [Batch, Channels]

        # Fully connected layer to get [Batch, num_classes]
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        logits = self.fc3(x)


        return logits  # Return raw logits
