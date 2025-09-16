"""
ChillAdam Training Script

A modular training script for ResNet architectures with configurable optimizers.
Supports ResNet-18 and ResNet-50 implemented from scratch.
Supports streaming of multiple datasets from Hugging Face: Tiny ImageNet, ImageNet-1k, Food-101, STL-10.
Supports multiple optimizers: ChillAdam, Adam, AdamW, SGD, RMSprop, Adamax, NAdam, RAdam.

Usage:
    # Default ChillAdam optimizer
    python main.py --model resnet18 --dataset tiny-imagenet --epochs 10 --batch-size 64
    
    # Adam optimizer
    python main.py --optimizer adam --lr 0.001 --model resnet50 --dataset imagenet-1k --epochs 20
    
    # SGD with momentum
    python main.py --optimizer sgd --lr 0.01 --momentum 0.9 --model resnet18 --dataset food101 --epochs 15
    
    # AdamW with weight decay
    python main.py --optimizer adamw --lr 0.002 --weight-decay 0.01 --model resnet50 --dataset stl10 --epochs 25
"""

import sys
import os

# Add the chilladam package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chilladam'))

from chilladam import (
    ChillAdam, create_scheduler, resnet18, resnet50,
    standard_se_resnet18, standard_se_resnet34, standard_se_resnet50,
    yat_resnet18, yat_resnet34, yat_resnet50,
    yat_resnet18_no_se, yat_resnet34_no_se, yat_resnet50_no_se,
    yat_se_resnet18, yat_se_resnet34, yat_se_resnet50,
    vit_base, vit_large
)
from chilladam.optimizers import create_optimizer
from chilladam.data import get_data_loaders
from chilladam.training import Trainer
from chilladam.config import parse_args, print_config


def create_model(model_name, num_classes, input_size):
    """
    Create the specified model.
    
    Arguments:
        model_name: model architecture name
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
    elif model_name == "standard_se_resnet18":
        print("Creating Standard SE-ResNet-18 model from scratch...")
        return standard_se_resnet18(num_classes=num_classes)
    elif model_name == "standard_se_resnet34":
        print("Creating Standard SE-ResNet-34 model from scratch...")
        return standard_se_resnet34(num_classes=num_classes)
    elif model_name == "standard_se_resnet50":
        print("Creating Standard SE-ResNet-50 model from scratch...")
        return standard_se_resnet50(num_classes=num_classes)
    elif model_name == "yat_resnet18":
        print("Creating YAT-ResNet-18 model from scratch...")
        return yat_resnet18(num_classes=num_classes)
    elif model_name == "yat_resnet34":
        print("Creating YAT-ResNet-34 model from scratch...")
        return yat_resnet34(num_classes=num_classes)
    elif model_name == "yat_resnet50":
        print("Creating YAT-ResNet-50 model from scratch...")
        return yat_resnet50(num_classes=num_classes)
    elif model_name == "yat_resnet18_no_se":
        print("Creating YAT-ResNet-18 model (no SE) from scratch...")
        return yat_resnet18_no_se(num_classes=num_classes)
    elif model_name == "yat_resnet34_no_se":
        print("Creating YAT-ResNet-34 model (no SE) from scratch...")
        return yat_resnet34_no_se(num_classes=num_classes)
    elif model_name == "yat_resnet50_no_se":
        print("Creating YAT-ResNet-50 model (no SE) from scratch...")
        return yat_resnet50_no_se(num_classes=num_classes)
    elif model_name == "yat_se_resnet18":
        print("Creating YAT SE-ResNet-18 model from scratch...")
        return yat_se_resnet18(num_classes=num_classes)
    elif model_name == "yat_se_resnet34":
        print("Creating YAT SE-ResNet-34 model from scratch...")
        return yat_se_resnet34(num_classes=num_classes)
    elif model_name == "yat_se_resnet50":
        print("Creating YAT SE-ResNet-50 model from scratch...")
        return yat_se_resnet50(num_classes=num_classes)
    elif model_name == "vit_base":
        print("Creating Vision Transformer Base model from scratch...")
        return vit_base(num_classes=num_classes, img_size=input_size)
    elif model_name == "vit_large":
        print("Creating Vision Transformer Large model from scratch...")
        return vit_large(num_classes=num_classes, img_size=input_size)
    else:
        supported_models = [
            "resnet18", "resnet50", 
            "standard_se_resnet18", "standard_se_resnet34", "standard_se_resnet50",
            "yat_resnet18", "yat_resnet34", "yat_resnet50", 
            "yat_resnet18_no_se", "yat_resnet34_no_se", "yat_resnet50_no_se",
            "yat_se_resnet18", "yat_se_resnet34", "yat_se_resnet50",
            "vit_base", "vit_large"
        ]
        raise ValueError(f"Unknown model: {model_name}. Choose from: {supported_models}")


def main():
    """Main training function."""
    try:
        # Parse configuration
        config = parse_args()
        print_config(config)
        
        # Create data loaders
        print(f"\nPreparing {config.dataset} data loaders...")
        train_dataloader, val_dataloader = get_data_loaders(
            dataset_name=config.dataset,
            batch_size=config.batch_size,
            image_size=config.image_size,
            shuffle_buffer_size=config.shuffle_buffer_size
        )
        
        # Create model
        print(f"\nInitializing {config.model_name.upper()} model...")
        model = create_model(
            model_name=config.model_name,
            num_classes=config.num_classes,
            input_size=config.image_size
        )
        
        # Create optimizer
        print(f"Setting up {config.optimizer.upper()} optimizer...")
        optimizer = create_optimizer(
            optimizer_name=config.optimizer,
            model_parameters=model.parameters(),
            lr=config.lr,
            min_lr=config.min_lr,
            max_lr=config.max_lr,
            eps=config.eps,
            betas=config.betas,
            weight_decay=config.weight_decay,
            l1_lambda=config.l1_lambda,
            momentum=config.momentum,
            alpha=config.alpha
        )
        
        # Create scheduler
        scheduler = None
        if config.use_scheduler and config.scheduler != "none":
            print(f"Setting up {config.scheduler.upper()} learning rate scheduler...")
            
            scheduler_kwargs = {
                'scheduler_name': config.scheduler,
                'optimizer': optimizer,
                'eta_min': config.eta_min
            }
            
            if config.scheduler == "cosine_warmup":
                scheduler_kwargs.update({
                    'total_epochs': config.total_epochs,
                    'warmup_epochs': config.warmup_epochs,
                    'linear_decay_epochs': config.linear_decay_epochs,
                    'final_lr': config.final_lr,
                    'restart': config.restart
                })
                # Calculate phases for better logging
                cosine_epochs = config.total_epochs - config.warmup_epochs - config.linear_decay_epochs
                phases = []
                if config.warmup_epochs > 0:
                    phases.append(f"{config.warmup_epochs} warmup")
                if cosine_epochs > 0:
                    phases.append(f"{cosine_epochs} cosine")
                if config.linear_decay_epochs > 0:
                    phases.append(f"{config.linear_decay_epochs} linear decay")
                
                phase_desc = " + ".join(phases) + f" epochs"
                restart_desc = " (with restart)" if config.restart else " (no restart)"
                print(f"Cosine warmup scheduler: {phase_desc}{restart_desc}")
                
                if config.linear_decay_epochs > 0:
                    final_lr_val = config.final_lr if config.final_lr is not None else config.eta_min
                    print(f"  Linear decay: {config.eta_min} → {final_lr_val}")
            else:
                scheduler_kwargs['t_max'] = config.t_max
                print(f"Scheduler will run for {config.t_max} epochs with eta_min={config.eta_min}")
            
            scheduler = create_scheduler(**scheduler_kwargs)
        else:
            print("No learning rate scheduler enabled")
        
        # Create trainer
        print(f"Initializing trainer with device: {config.device}")
        if config.l1_lambda > 0:
            print(f"L1 regularization enabled with lambda = {config.l1_lambda}")
        
        # Prepare wandb config if using wandb
        wandb_config = None
        if config.use_wandb:
            wandb_config = {
                'project': config.wandb_project,
                'run_name': config.wandb_run_name,
                'model': config.model_name,
                'dataset': config.dataset,
                'optimizer': config.optimizer,
                'epochs': config.num_epochs,
                'batch_size': config.batch_size,
                'learning_rate': config.lr if config.optimizer != 'chilladam' else config.max_lr,
                'min_lr': config.min_lr if config.optimizer == 'chilladam' else None,
                'max_lr': config.max_lr if config.optimizer == 'chilladam' else None,
                'momentum': config.momentum if config.optimizer in ['sgd', 'rmsprop'] else None,
                'alpha': config.alpha if config.optimizer == 'rmsprop' else None,
                'weight_decay': config.weight_decay,
                'l1_lambda': config.l1_lambda if config.optimizer in ['chilladam', 'chillsgd'] else None,
                'image_size': config.image_size,
                'num_classes': config.num_classes,
                'device': config.device,
                'use_scheduler': config.use_scheduler,
                'scheduler': config.scheduler if config.use_scheduler else None,
                't_max': config.t_max if config.use_scheduler and config.scheduler in ['cosine'] else None,
                'eta_min': config.eta_min if config.use_scheduler and config.scheduler in ['cosine', 'cosine_warmup'] else None,
                'total_epochs': config.total_epochs if config.use_scheduler and config.scheduler == 'cosine_warmup' else None,
                'warmup_epochs': config.warmup_epochs if config.use_scheduler and config.scheduler == 'cosine_warmup' else None,
                'linear_decay_epochs': config.linear_decay_epochs if config.use_scheduler and config.scheduler == 'cosine_warmup' and config.linear_decay_epochs > 0 else None,
                'final_lr': config.final_lr if config.use_scheduler and config.scheduler == 'cosine_warmup' and config.final_lr is not None else None,
                'restart': config.restart if config.use_scheduler and config.scheduler == 'cosine_warmup' else None,
                'wandb_watch': config.wandb_watch,
                'wandb_watch_log_freq': config.wandb_watch_log_freq
            }
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.device,
            l1_lambda=config.l1_lambda,
            use_wandb=config.use_wandb,
            wandb_config=wandb_config,
            wandb_watch=config.wandb_watch,
            wandb_watch_log_freq=config.wandb_watch_log_freq
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
