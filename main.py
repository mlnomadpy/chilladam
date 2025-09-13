"""
ChillAdam Training Script

A modular training script for ResNet architectures with ChillAdam optimizer.
Supports ResNet-18 and ResNet-50 implemented from scratch.

Usage:
    python main.py --model resnet18 --epochs 10 --batch-size 64
    python main.py --model resnet50 --epochs 20 --batch-size 32
"""

import sys
import os

# Add the chilladam package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chilladam'))

from chilladam import ChillAdam, resnet18, resnet50
from chilladam.data import get_tiny_imagenet_loaders
from chilladam.training import Trainer
from chilladam.config import parse_args, print_config


def create_model(model_name, num_classes, input_size):
    """
    Create the specified ResNet model.
    
    Arguments:
        model_name: 'resnet18' or 'resnet50'
        num_classes: number of output classes
        input_size: input image size
        
    Returns:
        PyTorch model
    """
    if model_name == "resnet18":
        print("Creating ResNet-18 model from scratch...")
        return resnet18(num_classes=num_classes, input_size=input_size)
    elif model_name == "resnet50":
        print("Creating ResNet-50 model from scratch...")
        return resnet50(num_classes=num_classes, input_size=input_size)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'resnet18' or 'resnet50'")


def main():
    """Main training function."""
    try:
        # Parse configuration
        config = parse_args()
        print_config(config)
        
        # Create data loaders
        print("\nPreparing data loaders...")
        train_dataloader, val_dataloader = get_tiny_imagenet_loaders(
            batch_size=config.batch_size,
            image_size=config.image_size
        )
        
        # Create model
        print(f"\nInitializing {config.model_name.upper()} model...")
        model = create_model(
            model_name=config.model_name,
            num_classes=config.num_classes,
            input_size=config.image_size
        )
        
        # Create optimizer
        print("Setting up ChillAdam optimizer...")
        optimizer = ChillAdam(
            model.parameters(),
            min_lr=config.min_lr,
            max_lr=config.max_lr,
            eps=config.eps,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
        
        # Create trainer
        print(f"Initializing trainer with device: {config.device}")
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=config.device
        )
        
        # Start training
        print(f"\nStarting training for {config.num_epochs} epochs...")
        trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=config.num_epochs
        )
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
