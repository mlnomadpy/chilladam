"""
ResNet implementation from scratch.

Supports ResNet-18 and ResNet-50 architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_layer_norm_2d(x, layer_norm):
    """
    Apply LayerNorm to 2D feature maps by permuting dimensions.
    
    Args:
        x: Input tensor of shape (N, C, H, W)
        layer_norm: LayerNorm layer expecting (N, H, W, C)
    
    Returns:
        Output tensor of shape (N, C, H, W)
    """
    # Permute from (N, C, H, W) to (N, H, W, C) for LayerNorm
    b, c, h, w = x.size()
    x = x.permute(0, 2, 3, 1)  # (N, H, W, C)
    x = layer_norm(x)          # Apply LayerNorm over channel dimension
    x = x.permute(0, 3, 1, 2)  # Back to (N, C, H, W)
    return x


class BasicBlock(nn.Module):
    """
    Basic block for ResNet-18 and ResNet-34.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.ln1 = nn.LayerNorm(planes, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.ln2 = nn.LayerNorm(planes, bias=False)

        self.shortcut = nn.Sequential()
        self.shortcut_ln = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False)
            )
            self.shortcut_ln = nn.LayerNorm(self.expansion * planes, bias=False)

    def forward(self, x):
        out = F.relu(apply_layer_norm_2d(self.conv1(x), self.ln1))
        out = apply_layer_norm_2d(self.conv2(out), self.ln2)
        
        shortcut_out = self.shortcut(x)
        if self.shortcut_ln is not None:
            shortcut_out = apply_layer_norm_2d(shortcut_out, self.shortcut_ln)
        
        out += shortcut_out
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet-50, ResNet-101, and ResNet-152.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.ln1 = nn.LayerNorm(planes, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.ln2 = nn.LayerNorm(planes, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.ln3 = nn.LayerNorm(self.expansion * planes, bias=False)

        self.shortcut = nn.Sequential()
        self.shortcut_ln = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False)
            )
            self.shortcut_ln = nn.LayerNorm(self.expansion * planes, bias=False)

    def forward(self, x):
        out = F.relu(apply_layer_norm_2d(self.conv1(x), self.ln1))
        out = F.relu(apply_layer_norm_2d(self.conv2(out), self.ln2))
        out = apply_layer_norm_2d(self.conv3(out), self.ln3)
        
        shortcut_out = self.shortcut(x)
        if self.shortcut_ln is not None:
            shortcut_out = apply_layer_norm_2d(shortcut_out, self.shortcut_ln)
        
        out += shortcut_out
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet architecture implementation.
    
    Arguments:
        block: BasicBlock for ResNet-18/34, Bottleneck for ResNet-50/101/152
        num_blocks: list of number of blocks in each layer
        num_classes: number of output classes (default: 1000)
        input_size: input image size, used to adapt conv1 layer (default: 224)
    """
    
    def __init__(self, block, num_blocks, num_classes=1000, input_size=224):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        # Adapt first conv layer based on input size
        if input_size == 64:  # Tiny ImageNet
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()  # No maxpool for small images
        else:  # ImageNet and larger
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
        self.ln1 = nn.LayerNorm(64, bias=False)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a layer with the specified number of blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(apply_layer_norm_2d(self.conv1(x), self.ln1))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out


def resnet18(num_classes=1000, input_size=224):
    """
    ResNet-18 model.
    
    Arguments:
        num_classes: number of output classes
        input_size: input image size (64 for Tiny ImageNet, 224 for ImageNet)
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_size=input_size)


def resnet50(num_classes=1000, input_size=224):
    """
    ResNet-50 model.
    
    Arguments:
        num_classes: number of output classes
        input_size: input image size (64 for Tiny ImageNet, 224 for ImageNet)
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, input_size=input_size)