"""
Configuration management for ChillAdam training.
"""

import argparse
import torch


class Config:
    """Configuration class for training parameters."""
    
    def __init__(self):
        # Default configuration
        self.dataset = "tiny-imagenet"  # Default dataset
        self.num_classes = 200  # Will be set based on dataset
        self.batch_size = 64
        self.num_epochs = 10
        self.learning_rate = 1e-3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = 64  # Will be set based on dataset
        
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
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, 
                       choices=["tiny-imagenet", "imagenet-1k", "food101", "stl10"],
                       default="tiny-imagenet", 
                       help="Dataset to use for training")
    
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
    parser.add_argument("--image-size", type=int, default=None,
                       help="Image size for resizing (auto-detected based on dataset if not specified)")
    
    args = parser.parse_args()
    
    # Create config object
    config = Config()
    
    # Update config with parsed arguments
    config.model_name = args.model
    config.dataset = args.dataset
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.min_lr = args.min_lr
    config.max_lr = args.max_lr
    config.weight_decay = args.weight_decay
    
    # Set dataset-specific defaults
    dataset_configs = {
        "tiny-imagenet": {"num_classes": 200, "image_size": 64},
        "imagenet-1k": {"num_classes": 1000, "image_size": 224},
        "food101": {"num_classes": 101, "image_size": 224},
        "stl10": {"num_classes": 10, "image_size": 96}
    }
    
    if config.dataset in dataset_configs:
        config.num_classes = dataset_configs[config.dataset]["num_classes"]
        default_image_size = dataset_configs[config.dataset]["image_size"]
    else:
        # Fallback defaults
        config.num_classes = 200
        default_image_size = 64
    
    # Override image size if specified
    config.image_size = args.image_size if args.image_size is not None else default_image_size
    
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
    print(f"Dataset: {config.dataset}")
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