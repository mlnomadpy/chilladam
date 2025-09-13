"""
Squeeze-and-Excitation (SE) and YAT-based model implementations.

This module provides ResNet-like architectures with SE blocks and YAT (Yet Another Transformation) layers.
Includes both standard SE implementations and YAT-based variants using the nmn.torch package.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nmn.torch.nmn import YatNMN
from nmn.torch.conv import YatConv2d


class SELayer(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class YatSELayer(nn.Module):
    """Squeeze-and-Excitation block with YAT transformation."""
    def __init__(self, channel, reduction=16):
        super(YatSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            YatNMN(channel, channel // reduction, bias=False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicStandardBlock(nn.Module):
    """A basic residual block for the StandardConvNet, inspired by ResNet, with Squeeze-and-Excitation."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicStandardBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Apply Squeeze-and-Excitation
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicYATBlock(nn.Module):
    """A basic residual block for the YATConvNet, with Squeeze-and-Excitation."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_alpha=True, use_dropconnect=False, drop_rate=0.1):
        super(BasicYATBlock, self).__init__()
        self.yat_conv = YatConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                  use_alpha=use_alpha, use_dropconnect=use_dropconnect,
                                  drop_rate=drop_rate, bias=False, epsilon=0.007)
        self.lin_conv = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.se = YatSELayer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.yat_conv(x, deterministic=not self.training)
        out = self.lin_conv(out)
        out = self.se(out) # Apply Squeeze-and-Excitation
        out += identity
        return out


class BasicYATBlockNoSE(nn.Module):
    """A basic residual block for the YATConvNet, without Squeeze-and-Excitation, with LayerNorm after skip connection."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_alpha=True, use_dropconnect=False, drop_rate=0.1):
        super(BasicYATBlockNoSE, self).__init__()
        self.yat_conv = YatConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                  use_alpha=use_alpha, use_dropconnect=use_dropconnect,
                                  drop_rate=drop_rate, bias=False, epsilon=0.007)
        self.lin_conv = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        # LayerNorm for post-skip connection normalization
        # Normalize over channel dimension for 2D feature maps
        self.layer_norm = nn.LayerNorm(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.yat_conv(x, deterministic=not self.training)
        out = self.lin_conv(out)
        out += identity
        
        # Apply LayerNorm after skip connection
        # LayerNorm expects input of shape (N, ..., normalized_shape)
        # For 2D feature maps (N, C, H, W), we need to permute to (N, H, W, C) for channel normalization
        b, c, h, w = out.size()
        out = out.permute(0, 2, 3, 1)  # (N, H, W, C)
        out = self.layer_norm(out)     # Apply LayerNorm over channel dimension
        out = out.permute(0, 3, 1, 2)  # Back to (N, C, H, W)
        
        return out


class StandardConvNet(nn.Module):
    """A standard CNN with a ResNet-like architecture."""
    def __init__(self, block, num_blocks, num_classes=10):
        super(StandardConvNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class YATConvNet(nn.Module):
    """A YAT-based CNN with a ResNet-like architecture."""
    def __init__(self, block, num_blocks, num_classes=200, use_alpha=True, use_dropconnect=False, drop_rate=0.1):
        super(YATConvNet, self).__init__()
        self.in_planes = 64
        self.use_alpha = use_alpha
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc_yat = nn.Linear(512 * block.expansion, num_classes, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s,
                                use_alpha=self.use_alpha,
                                use_dropconnect=self.use_dropconnect,
                                drop_rate=self.drop_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc_yat(out)
        return out


# Factory functions for creating predefined models
def standard_se_resnet18(num_classes=10):
    """
    Standard ResNet-18 with Squeeze-and-Excitation blocks.
    
    Arguments:
        num_classes: number of output classes
    """
    return StandardConvNet(BasicStandardBlock, [2, 2, 2, 2], num_classes=num_classes)


def standard_se_resnet34(num_classes=10):
    """
    Standard ResNet-34 with Squeeze-and-Excitation blocks.
    
    Arguments:
        num_classes: number of output classes
    """
    return StandardConvNet(BasicStandardBlock, [3, 4, 6, 3], num_classes=num_classes)


def yat_resnet18(num_classes=200, use_alpha=True, use_dropconnect=False, drop_rate=0.1):
    """
    YAT-based ResNet-18 with Squeeze-and-Excitation blocks.
    
    Arguments:
        num_classes: number of output classes
        use_alpha: whether to use alpha scaling in YAT layers
        use_dropconnect: whether to use DropConnect in YAT layers
        drop_rate: dropout rate for DropConnect
    """
    return YATConvNet(BasicYATBlock, [2, 2, 2, 2], 
                      num_classes=num_classes,
                      use_alpha=use_alpha,
                      use_dropconnect=use_dropconnect,
                      drop_rate=drop_rate)


def yat_resnet34(num_classes=200, use_alpha=True, use_dropconnect=False, drop_rate=0.1):
    """
    YAT-based ResNet-34 with Squeeze-and-Excitation blocks.
    
    Arguments:
        num_classes: number of output classes
        use_alpha: whether to use alpha scaling in YAT layers
        use_dropconnect: whether to use DropConnect in YAT layers
        drop_rate: dropout rate for DropConnect
    """
    return YATConvNet(BasicYATBlock, [3, 4, 6, 3], 
                      num_classes=num_classes,
                      use_alpha=use_alpha,
                      use_dropconnect=use_dropconnect,
                      drop_rate=drop_rate)


def yat_resnet18_no_se(num_classes=200, use_alpha=True, use_dropconnect=False, drop_rate=0.1):
    """
    YAT-based ResNet-18 without Squeeze-and-Excitation blocks, with LayerNorm after skip connection.
    
    Arguments:
        num_classes: number of output classes
        use_alpha: whether to use alpha scaling in YAT layers
        use_dropconnect: whether to use DropConnect in YAT layers
        drop_rate: dropout rate for DropConnect
    """
    return YATConvNet(BasicYATBlockNoSE, [2, 2, 2, 2], 
                      num_classes=num_classes,
                      use_alpha=use_alpha,
                      use_dropconnect=use_dropconnect,
                      drop_rate=drop_rate)


def yat_resnet34_no_se(num_classes=200, use_alpha=True, use_dropconnect=False, drop_rate=0.1):
    """
    YAT-based ResNet-34 without Squeeze-and-Excitation blocks, with LayerNorm after skip connection.
    
    Arguments:
        num_classes: number of output classes
        use_alpha: whether to use alpha scaling in YAT layers
        use_dropconnect: whether to use DropConnect in YAT layers
        drop_rate: dropout rate for DropConnect
    """
    return YATConvNet(BasicYATBlockNoSE, [3, 4, 6, 3], 
                      num_classes=num_classes,
                      use_alpha=use_alpha,
                      use_dropconnect=use_dropconnect,
                      drop_rate=drop_rate)