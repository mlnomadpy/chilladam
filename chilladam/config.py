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
        
        # Optimizer configuration
        self.optimizer = "chilladam"  # Default optimizer
        
        # Standard optimizer parameters
        self.lr = 1e-3  # Standard learning rate for most optimizers
        self.momentum = 0.9  # For SGD and RMSprop
        self.alpha = 0.99  # For RMSprop
        
        # ChillAdam specific parameters (only used when optimizer is chilladam)
        self.min_lr = 1e-5
        self.max_lr = 1.0
        self.eps = 1e-8
        self.betas = (0.9, 0.999)
        self.weight_decay = 0
        self.l1_lambda = 0  # L1 regularization strength (Lasso penalty)
        
        # Data loading parameters
        self.shuffle_buffer_size = 10000  # Increased from default 1000 to better mix classes
        
        # Wandb configuration
        self.use_wandb = False
        self.wandb_project = "chilladam-training"
        self.wandb_run_name = None
        self.wandb_watch = False
        self.wandb_watch_log_freq = 100


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Config: configuration object with parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train neural networks with configurable optimizer")
    
    # Model arguments
    parser.add_argument("--model", type=str, 
                       choices=["resnet18", "resnet50", 
                               "standard_se_resnet18", "standard_se_resnet34", "standard_se_resnet50",
                               "yat_resnet18", "yat_resnet34", "yat_resnet50",
                               "yat_resnet18_no_se", "yat_resnet34_no_se", "yat_resnet50_no_se",
                               "yat_se_resnet18", "yat_se_resnet34", "yat_se_resnet50"], 
                       default="resnet18", help="Model architecture to use")
    
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
    parser.add_argument("--optimizer", type=str, 
                       choices=["chilladam", "chillsgd", "adam", "adamw", "sgd", "rmsprop", "adamax", "nadam", "radam"],
                       default="chilladam", 
                       help="Optimizer to use for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate for standard optimizers (not used for ChillAdam)")
    parser.add_argument("--momentum", type=float, default=0.9,
                       help="Momentum for SGD and RMSprop optimizers")
    parser.add_argument("--alpha", type=float, default=0.99,
                       help="Alpha parameter for RMSprop optimizer")
    
    # ChillAdam and ChillSGD specific arguments
    parser.add_argument("--min-lr", type=float, default=1e-5,
                       help="Minimum learning rate for ChillAdam and ChillSGD")
    parser.add_argument("--max-lr", type=float, default=1.0,
                       help="Maximum learning rate for ChillAdam and ChillSGD")
    parser.add_argument("--weight-decay", type=float, default=0,
                       help="Weight decay for optimizers")
    parser.add_argument("--l1-lambda", type=float, default=0,
                       help="L1 regularization strength (Lasso penalty) for ChillAdam and ChillSGD")
    
    # Dataset arguments
    parser.add_argument("--image-size", type=int, default=None,
                       help="Image size for resizing (auto-detected based on dataset if not specified)")
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000,
                       help="Buffer size for shuffling streaming datasets (larger values better mix classes)")
    
    # Wandb arguments
    parser.add_argument("--use-wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="chilladam-training",
                       help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                       help="Wandb run name (auto-generated if not specified)")
    parser.add_argument("--wandb-watch", action="store_true",
                       help="Enable wandb.watch() to log model gradients and parameters")
    parser.add_argument("--wandb-watch-log-freq", type=int, default=100,
                       help="Log frequency for wandb.watch() (default: 100)")
    
    args = parser.parse_args()
    
    # Create config object
    config = Config()
    
    # Update config with parsed arguments
    config.model_name = args.model
    config.dataset = args.dataset
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.optimizer = args.optimizer
    config.lr = args.lr
    config.momentum = args.momentum
    config.alpha = args.alpha
    config.min_lr = args.min_lr
    config.max_lr = args.max_lr
    config.weight_decay = args.weight_decay
    config.l1_lambda = args.l1_lambda
    
    # Wandb configuration
    config.use_wandb = args.use_wandb
    config.wandb_project = args.wandb_project
    config.wandb_run_name = args.wandb_run_name
    config.wandb_watch = args.wandb_watch
    config.wandb_watch_log_freq = args.wandb_watch_log_freq
    
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
    
    # Set shuffle buffer size
    config.shuffle_buffer_size = args.shuffle_buffer_size
    
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
    print(f"Optimizer: {config.optimizer.upper()}")
    print(f"Shuffle Buffer Size: {config.shuffle_buffer_size}")

    if config.optimizer in ["chilladam", "chillsgd"]:
        print(f"{config.optimizer.upper()} Min LR: {config.min_lr}")
        print(f"{config.optimizer.upper()} Max LR: {config.max_lr}")
        if config.l1_lambda > 0:
            print(f"L1 Regularization (Lasso): {config.l1_lambda}")
    else:
        print(f"Learning Rate: {config.lr}")
        if config.optimizer == "sgd":
            print(f"Momentum: {config.momentum}")
        elif config.optimizer == "rmsprop":
            print(f"Momentum: {config.momentum}")
            print(f"Alpha: {config.alpha}")
    print(f"Weight Decay: {config.weight_decay}")
    print(f"Use Wandb: {config.use_wandb}")
    if config.use_wandb:
        print(f"Wandb Project: {config.wandb_project}")
        if config.wandb_run_name:
            print(f"Wandb Run Name: {config.wandb_run_name}")
        print(f"Wandb Watch: {config.wandb_watch}")
        if config.wandb_watch:
            print(f"Wandb Watch Log Freq: {config.wandb_watch_log_freq}")
    print("=" * 50)
