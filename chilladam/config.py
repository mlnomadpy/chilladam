"""
Configuration management for ChillAdam training.
"""

import argparse
import torch


class Config:
    """Configuration class for training parameters."""
    
    def __init__(self):
        # Default configuration
        self.num_classes = 200  # Tiny ImageNet has 200 classes
        self.batch_size = 64
        self.num_epochs = 10
        self.learning_rate = 1e-3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = 64  # Tiny ImageNet image size
        
        # Model configuration
        self.model_name = "resnet18"  # Default to ResNet-18
        
        # ChillAdam optimizer parameters
        self.min_lr = 1e-5
        self.max_lr = 1.0
        self.eps = 1e-8
        self.betas = (0.9, 0.999)
        self.weight_decay = 0


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Config: configuration object with parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train ResNet with ChillAdam optimizer")
    
    # Model arguments
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet50"], 
                       default="resnet18", help="ResNet architecture to use")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for training and validation")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], 
                       default=None, help="Device to use for training")
    
    # Optimizer arguments
    parser.add_argument("--min-lr", type=float, default=1e-5,
                       help="Minimum learning rate for ChillAdam")
    parser.add_argument("--max-lr", type=float, default=1.0,
                       help="Maximum learning rate for ChillAdam")
    parser.add_argument("--weight-decay", type=float, default=0,
                       help="Weight decay for ChillAdam")
    
    # Dataset arguments
    parser.add_argument("--image-size", type=int, default=64,
                       help="Image size for resizing")
    
    args = parser.parse_args()
    
    # Create config object
    config = Config()
    
    # Update config with parsed arguments
    config.model_name = args.model
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.min_lr = args.min_lr
    config.max_lr = args.max_lr
    config.weight_decay = args.weight_decay
    config.image_size = args.image_size
    
    # Set device
    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"
    
    return config


def print_config(config):
    """
    Print configuration settings.
    
    Arguments:
        config: Config object to print
    """
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Image Size: {config.image_size}")
    print(f"Number of Classes: {config.num_classes}")
    print(f"ChillAdam Min LR: {config.min_lr}")
    print(f"ChillAdam Max LR: {config.max_lr}")
    print(f"Weight Decay: {config.weight_decay}")
    print("=" * 50)