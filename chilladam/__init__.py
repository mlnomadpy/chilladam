"""
ChillAdam: A modular deep learning library with custom optimizer and ResNet implementation.
"""

__version__ = "0.1.0"

from .optimizers import ChillAdam
from .models import ResNet, resnet18, resnet50

__all__ = [
    "ChillAdam",
    "ResNet", 
    "resnet18",
    "resnet50",
]