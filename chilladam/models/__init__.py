"""
Model architectures for ChillAdam library.
"""

from .resnet import ResNet, resnet18, resnet50
from .se_models import (
    SELayer, YatSELayer, BasicStandardBlock, BasicYATBlock, BasicYATBlockNoSE,
    BottleneckStandardBlock, BottleneckYATBlock, BottleneckYATBlockNoSE,
    StandardConvNet, YATConvNet,
    # New improved naming
    se_resnet18, se_resnet34, se_resnet50,
    yat_se_resnet18, yat_se_resnet34, yat_se_resnet50,
    yat_resnet18_plain, yat_resnet34_plain, yat_resnet50_plain,
    # Backward compatibility (current naming)
    standard_se_resnet18, standard_se_resnet34, standard_se_resnet50,
    yat_resnet18, yat_resnet34, yat_resnet50,
    yat_resnet18_no_se, yat_resnet34_no_se, yat_resnet50_no_se
)
from .vit import (
    VisionTransformer, PatchEmbedding, MultiHeadAttention, MLP, TransformerBlock,
    vit_base, vit_large
)

__all__ = [
    "ResNet", "resnet18", "resnet50",
    "SELayer", "YatSELayer", "BasicStandardBlock", "BasicYATBlock", "BasicYATBlockNoSE",
    "BottleneckStandardBlock", "BottleneckYATBlock", "BottleneckYATBlockNoSE",
    "StandardConvNet", "YATConvNet",
    # New improved naming
    "se_resnet18", "se_resnet34", "se_resnet50",
    "yat_se_resnet18", "yat_se_resnet34", "yat_se_resnet50",
    "yat_resnet18_plain", "yat_resnet34_plain", "yat_resnet50_plain",
    # Backward compatibility (current naming)
    "standard_se_resnet18", "standard_se_resnet34", "standard_se_resnet50",
    "yat_resnet18", "yat_resnet34", "yat_resnet50",
    "yat_resnet18_no_se", "yat_resnet34_no_se", "yat_resnet50_no_se",
    # Vision Transformer models
    "VisionTransformer", "PatchEmbedding", "MultiHeadAttention", "MLP", "TransformerBlock",
    "vit_base", "vit_large"
]