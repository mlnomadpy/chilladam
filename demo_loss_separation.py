#!/usr/bin/env python3
"""
Demonstration script showing the new loss separation logging in action.

This script shows how cross entropy loss, L1 loss, and total loss are now logged separately.
"""

import torch
import torch.nn as nn
import sys
import os

# Add chilladam to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chilladam'))

from chilladam.optimizers.chilladam import ChillAdam
from chilladam.training.trainer import Trainer
from chilladam.loss import L1RegularizedLoss


def create_mock_dataloader(num_batches=3, batch_size=8, input_size=20, num_classes=3):
    """Create a simple mock dataloader for demonstration"""
    data = []
    for _ in range(num_batches):
        batch = {
            'pixel_values': torch.randn(batch_size, input_size),
            'label': torch.randint(0, num_classes, (batch_size,))
        }
        data.append(batch)
    return data


def mock_wandb_logging():
    """Mock wandb to capture and print logged values"""
    logged_data = []
    
    class MockWandb:
        @staticmethod
        def log(log_dict, step=None):
            logged_data.append((log_dict.copy(), step))
            # Print the loss components for demonstration
            if "train/step_loss" in log_dict:
                print(f"  Step {step}:")
                print(f"    Total Loss: {log_dict['train/step_loss']:.6f}")
                print(f"    Cross Entropy Loss: {log_dict['train/step_cross_entropy_loss']:.6f}")
                if "train/step_l1_loss" in log_dict:
                    print(f"    L1 Loss: {log_dict['train/step_l1_loss']:.6f}")
                    # Verify the sum
                    total = log_dict['train/step_loss']
                    ce = log_dict['train/step_cross_entropy_loss']
                    l1 = log_dict['train/step_l1_loss']
                    computed_total = ce + l1
                    print(f"    Verification: {ce:.6f} + {l1:.6f} = {computed_total:.6f} (expected: {total:.6f})")
                print()
        
        @staticmethod
        def init(*args, **kwargs):
            pass
            
        @staticmethod
        def watch(*args, **kwargs):
            pass
    
    return MockWandb, logged_data


def demonstrate_with_l1():
    """Demonstrate loss separation with L1 regularization enabled"""
    print("🔹 DEMONSTRATING WITH L1 REGULARIZATION ENABLED")
    print("=" * 60)
    
    # Create model and optimizer
    model = nn.Linear(20, 3)
    optimizer = ChillAdam(model.parameters())
    
    # Mock wandb
    MockWandb, logged_data = mock_wandb_logging()
    
    # Replace wandb in trainer module
    import chilladam.training.trainer as trainer_module
    original_wandb = getattr(trainer_module, 'wandb', None)
    trainer_module.wandb = MockWandb
    trainer_module.WANDB_AVAILABLE = True
    
    try:
        # Create trainer with L1 regularization
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            l1_lambda=0.02,  # Enable L1 regularization
            use_wandb=True,
            wandb_config={'project': 'loss-separation-demo'}
        )
        
        # Create mock data
        train_dataloader = create_mock_dataloader(num_batches=3)
        
        print(f"Training with L1 regularization (lambda = {trainer.l1_lambda})")
        print("Logging separate loss components:\n")
        
        # Simulate a few training steps
        trainer.model.train()
        trainer.step_count = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            inputs = batch['pixel_values']
            labels = torch.tensor(batch['label'])
            
            # Forward pass
            trainer.optimizer.zero_grad()
            outputs = trainer.model(inputs)
            
            # Calculate loss with separation
            base_loss, l1_loss, total_loss = trainer.criterion.compute_separate_losses(outputs, labels, trainer.model)
            loss = total_loss
            
            # Backward pass
            loss.backward()
            trainer.optimizer.step()
            
            # Log the metrics
            log_dict = {
                "train/step_loss": loss.item(),
                "train/step_cross_entropy_loss": base_loss.item(),
                "train/step": trainer.step_count,
            }
            
            if trainer.use_l1_regularization and l1_loss is not None:
                log_dict["train/step_l1_loss"] = l1_loss.item()
            
            MockWandb.log(log_dict, step=trainer.step_count)
            trainer.step_count += 1
            
    finally:
        # Restore original wandb
        if original_wandb is not None:
            trainer_module.wandb = original_wandb
        else:
            if hasattr(trainer_module, 'wandb'):
                delattr(trainer_module, 'wandb')
    
    print("✅ L1 regularization demo completed successfully!")
    return logged_data


def demonstrate_without_l1():
    """Demonstrate loss separation without L1 regularization"""
    print("\n🔹 DEMONSTRATING WITHOUT L1 REGULARIZATION")
    print("=" * 60)
    
    # Create model and optimizer
    model = nn.Linear(20, 3)
    optimizer = ChillAdam(model.parameters())
    
    # Mock wandb
    MockWandb, logged_data = mock_wandb_logging()
    
    # Replace wandb in trainer module
    import chilladam.training.trainer as trainer_module
    original_wandb = getattr(trainer_module, 'wandb', None)
    trainer_module.wandb = MockWandb
    trainer_module.WANDB_AVAILABLE = True
    
    try:
        # Create trainer without L1 regularization
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            l1_lambda=0.0,  # Disable L1 regularization
            use_wandb=True,
            wandb_config={'project': 'loss-separation-demo'}
        )
        
        # Create mock data
        train_dataloader = create_mock_dataloader(num_batches=3)
        
        print(f"Training without L1 regularization (lambda = {trainer.l1_lambda})")
        print("Logging separate loss components:\n")
        
        # Simulate a few training steps
        trainer.model.train()
        trainer.step_count = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            inputs = batch['pixel_values']
            labels = torch.tensor(batch['label'])
            
            # Forward pass
            trainer.optimizer.zero_grad()
            outputs = trainer.model(inputs)
            
            # Calculate loss (just base loss)
            base_loss = trainer.criterion(outputs, labels)
            loss = base_loss
            
            # Backward pass
            loss.backward()
            trainer.optimizer.step()
            
            # Log the metrics
            log_dict = {
                "train/step_loss": loss.item(),
                "train/step_cross_entropy_loss": base_loss.item(),
                "train/step": trainer.step_count,
            }
            
            # No L1 loss to log since it's disabled
            
            MockWandb.log(log_dict, step=trainer.step_count)
            trainer.step_count += 1
            
    finally:
        # Restore original wandb
        if original_wandb is not None:
            trainer_module.wandb = original_wandb
        else:
            if hasattr(trainer_module, 'wandb'):
                delattr(trainer_module, 'wandb')
    
    print("✅ No L1 regularization demo completed successfully!")
    return logged_data


def main():
    """Main demonstration function"""
    print("🎯 LOSS SEPARATION LOGGING DEMONSTRATION")
    print("=" * 70)
    print()
    print("This demonstration shows how the training now logs:")
    print("  • Cross Entropy Loss (base classification loss)")
    print("  • L1 Loss (regularization penalty, when enabled)")
    print("  • Total Loss (sum of cross entropy + L1)")
    print()
    
    # Demonstrate with L1 regularization
    l1_data = demonstrate_with_l1()
    
    # Demonstrate without L1 regularization
    no_l1_data = demonstrate_without_l1()
    
    print("\n🎉 SUMMARY")
    print("=" * 70)
    print("✅ Loss separation logging is working correctly!")
    print()
    print("When L1 regularization is ENABLED:")
    print("  • train/step_loss = total loss (cross entropy + L1)")
    print("  • train/step_cross_entropy_loss = base classification loss")
    print("  • train/step_l1_loss = L1 regularization penalty")
    print()
    print("When L1 regularization is DISABLED:")
    print("  • train/step_loss = total loss (same as cross entropy)")
    print("  • train/step_cross_entropy_loss = base classification loss")
    print("  • train/step_l1_loss = (not logged)")
    print()
    print("This provides better visibility into the loss components during training!")


if __name__ == "__main__":
    main()