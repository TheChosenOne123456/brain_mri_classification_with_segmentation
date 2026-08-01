import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple3DCNN(nn.Module):
    has_classification_head = True
    has_subtype_head = False
    has_segmentation_head = False
    uses_capability_interface = False

    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool3d(2)
        self.gap = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.gap(x)          # [B, 64, 1, 1, 1]
        x = x.view(x.size(0), -1)
        return self.fc(x)
