"""
model.py - Custom Residual Neural Network for CIFAR-10 Image Classification
Week 5 Task: Deep Learning Application in Data Science
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for adaptive channel-wise feature recalibration.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    Standard 2-layer Residual Convolutional Block:
    x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+) -> ReLU
    Identity/Shortcut projection is applied when stride != 1 or channels mismatch.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out, inplace=True)


class CustomResNetCIFAR(nn.Module):
    """
    Custom Deep Residual Neural Network optimized for CIFAR-10 (32x32 input resolution).
    Architecture:
      - Stem: 3x3 Conv (64 channels) + BatchNorm + ReLU
      - Stage 1: 2x Residual Blocks (64 channels, 32x32)
      - Stage 2: 2x Residual Blocks (128 channels, 16x16)
      - Stage 3: 2x Residual Blocks (256 channels, 8x8)
      - Stage 4: 2x Residual Blocks (512 channels, 4x4)
      - Head: Global Average Pooling (1x1) + Dropout(0.3) + Linear(512 -> 10)
    Total Trainable Parameters: ~11.2 Million
    """
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3, use_se: bool = True):
        super(CustomResNetCIFAR, self).__init__()
        self.in_channels = 64

        # Initial Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Residual Stages
        self.stage1 = self._make_layer(64, num_blocks=2, stride=1, use_se=use_se)
        self.stage2 = self._make_layer(128, num_blocks=2, stride=2, use_se=use_se)
        self.stage3 = self._make_layer(256, num_blocks=2, stride=2, use_se=use_se)
        self.stage4 = self._make_layer(512, num_blocks=2, stride=2, use_se=use_se)

        # Output Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(512, num_classes)

        # Weight Initialization using Kaiming (He) Normal
        self._initialize_weights()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int, use_se: bool) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=s, use_se=use_se))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits

if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 32, 32)
    model = CustomResNetCIFAR(num_classes=10)
    output = model(dummy_input)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created successfully. Output shape: {output.shape}")
    print(f"Total Trainable Parameters: {total_params:,}")
