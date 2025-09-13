"""
ChillAdam: A modular deep learning library with custom optimizer and ResNet implementation.
"""

__version__ = "0.1.0"

from .optimizers import ChillAdam
from .models import (
    ResNet, resnet18, resnet50,
    standard_se_resnet18, standard_se_resnet34,
    yat_resnet18, yat_resnet34,
    yat_resnet18_no_se, yat_resnet34_no_se
)

__all__ = [
    "ChillAdam",
    "ResNet", 
    "resnet18",
    "resnet50",
    "standard_se_resnet18",
    "standard_se_resnet34",
    "yat_resnet18",
    "yat_resnet34",
    "yat_resnet18_no_se",
    "yat_resnet34_no_se",
]