"""src/models/simple_cnn.py — Small CNN for interpretability demonstrations."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """3-layer CNN for 3×32×32 input, n_classes output.

    Designed to be interpretable: each convolutional block is accessible
    via explicit attributes so GradCAM can hook into any layer.
    """

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        # conv blocks (hookable by GradCAM)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool  = nn.MaxPool2d(2)
        self.gap   = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(128 * 4 * 4, n_classes)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # → (32, 16, 16)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # → (64, 8, 8)
        x = F.relu(self.bn3(self.conv3(x)))               # → (128, 8, 8)
        x = self.gap(x)                                    # → (128, 4, 4)
        x = self.drop(x.flatten(1))
        return self.fc(x)

    @property
    def last_conv(self) -> nn.Module:
        """Return the final conv layer (GradCAM target)."""
        return self.conv3
