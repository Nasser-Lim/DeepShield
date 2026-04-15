"""
Pure-PyTorch Xception backbone matching DeepfakeBench's checkpoint key layout.

DeepfakeBench saves Xception weights with `backbone.` prefix and uses these
exact module names: conv1, bn1, conv2, bn2, block1..block12, conv3, bn3,
conv4, bn4, last_linear. We rebuild the same architecture here so checkpoints
load with strict=True (no key remapping needed).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                               padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1,
                 start_with_relu=True, grow_first=True):
        super().__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        rep = []
        filters = in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        inp = x
        x = self.rep(x)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x = x + skip
        return x


class Xception(nn.Module):
    """
    Xception with DeepfakeBench's layer naming.
    last_linear: can be a plain Linear (Xception/SPSL) or
                 Sequential(Dropout, Linear) (F3Net) — configurable via head_type.
    in_channels: 3 for Xception, 4 for SPSL (RGB+phase), 12 for F3Net (4 DCT bands x 3 RGB).
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2, head_type: str = "linear"):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        if head_type == "linear":
            self.last_linear = nn.Linear(2048, num_classes)
        elif head_type == "dropout_linear":
            # F3Net uses Sequential(Dropout, Linear) — index '.1.weight' in checkpoint
            self.last_linear = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2048, num_classes),
            )
        else:
            raise ValueError(f"unknown head_type: {head_type}")

    def features(self, x):
        x = self.conv1(x); x = self.bn1(x); x = F.relu(x, inplace=True)
        x = self.conv2(x); x = self.bn2(x); x = F.relu(x, inplace=True)
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = self.block4(x); x = self.block5(x); x = self.block6(x); x = self.block7(x)
        x = self.block8(x); x = self.block9(x); x = self.block10(x); x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x); x = self.bn3(x); x = F.relu(x, inplace=True)
        x = self.conv4(x); x = self.bn4(x); x = F.relu(x, inplace=True)
        return x

    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.last_linear(x)

    def forward(self, x):
        return self.logits(self.features(x))


class DetectorNet(nn.Module):
    """
    Wraps Xception in `backbone.` attribute to match DeepfakeBench checkpoints.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2, head_type: str = "linear"):
        super().__init__()
        self.backbone = Xception(in_channels=in_channels, num_classes=num_classes,
                                 head_type=head_type)

    def forward(self, x):
        return self.backbone(x)
