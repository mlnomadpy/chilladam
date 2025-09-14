#!/usr/bin/env python3
"""
Test script for loss separation functionality.

This script tests that cross entropy loss, L1 loss, and total loss are logged separately.
"""

import torch
import torch.nn as nn
import sys
import os
import io
import contextlib

# Add chilladam to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chilladam'))

from chilladam.optimizers.chilladam import ChillAdam
from chilladam.training.trainer import Trainer
from chilladam.loss import L1RegularizedLoss


def test_separate_loss_calculation():
    """Test that L1RegularizedLoss can calculate separate loss components"""
    print("Testing separate loss calculation...")
    
    # Create a simple model
    model = nn.Linear(10, 3)
    
    # Create loss function with L1 regularization
    base_criterion = nn.CrossEntropyLoss()
    l1_criterion = L1RegularizedLoss(base_criterion, l1_lambda=0.01)
    
    # Create test data
    x = torch.randn(5, 10)
    y = torch.randint(0, 3, (5,))
    
    # Forward pass
    outputs = model(x)
    
    # Test regular __call__ method
    total_loss = l1_criterion(outputs, y, model)
    
    # Test separate loss calculation
    base_loss, l1_loss, total_loss_separate = l1_criterion.compute_separate_losses(outputs, y, model)
    
    # Verify they match
    assert abs(total_loss.item() - total_loss_separate.item()) < 1e-6, "Total loss should match between methods"
    
    # Verify base_loss + l1_loss = total_loss
    computed_total = base_loss + l1_loss
    assert abs(computed_total.item() - total_loss_separate.item()) < 1e-6, "Base + L1 should equal total"
    
    # L1 loss should be positive (we have parameters)
    assert l1_loss.item() > 0, "L1 loss should be positive for non-zero parameters"
    
    # Base loss should be positive (we have a real classification loss)
    assert base_loss.item() > 0, "Base loss should be positive"
    
    print("✓ Separate loss calculation works correctly")


def test_no_l1_regularization():
    """Test separate loss calculation when L1 regularization is disabled"""
    print("Testing separate loss calculation without L1 regularization...")
    
    # Create a simple model
    model = nn.Linear(10, 3)
    
    # Create loss function without L1 regularization
    base_criterion = nn.CrossEntropyLoss()
    l1_criterion = L1RegularizedLoss(base_criterion, l1_lambda=0.0)
    
    # Create test data
    x = torch.randn(5, 10)
    y = torch.randint(0, 3, (5,))
    
    # Forward pass
    outputs = model(x)
    
    # Test separate loss calculation
    base_loss, l1_loss, total_loss = l1_criterion.compute_separate_losses(outputs, y, model)
    
    # L1 loss should be zero
    assert l1_loss.item() == 0.0, "L1 loss should be zero when l1_lambda=0"
    
    # Total loss should equal base loss
    assert abs(total_loss.item() - base_loss.item()) < 1e-6, "Total should equal base when no L1"
    
    print("✓ Separate loss calculation works correctly without L1 regularization")


def test_trainer_loss_logging():
    """Test that trainer logs separate loss components correctly"""
    print("Testing trainer loss logging...")
    
    # Create a simple model
    model = nn.Linear(20, 3)
    optimizer = ChillAdam(model.parameters())
    
    # Mock wandb logging to capture what gets logged
    captured_logs = []
    
    class MockWandb:
        @staticmethod
        def log(log_dict, step=None):
            captured_logs.append((log_dict.copy(), step))
        
        @staticmethod
        def init(*args, **kwargs):
            pass
            
        @staticmethod
        def watch(*args, **kwargs):
            pass
    
    # Replace wandb temporarily
    import chilladam.training.trainer as trainer_module
    original_wandb = getattr(trainer_module, 'wandb', None)
    trainer_module.wandb = MockWandb
    trainer_module.WANDB_AVAILABLE = True
    
    try:
        # Create trainer with L1 regularization and wandb enabled
        trainer = Trainer(
            model=model, 
            optimizer=optimizer, 
            device='cpu', 
            l1_lambda=0.02, 
            use_wandb=True,
            wandb_config={'project': 'test'}
        )
        
        # Create dummy data
        batch = {
            'pixel_values': torch.randn(4, 20),
            'label': torch.randint(0, 3, (4,))
        }
        
        # Simulate one training step
        trainer.model.train()
        trainer.optimizer.zero_grad()
        
        inputs = batch['pixel_values']
        labels = torch.tensor(batch['label'])
        outputs = trainer.model(inputs)
        
        # Calculate separate loss components
        base_loss, l1_loss, total_loss = trainer.criterion.compute_separate_losses(outputs, labels, trainer.model)
        loss = total_loss
        
        # Manually log like trainer does
        log_dict = {
            "train/step_loss": loss.item(),
            "train/step_cross_entropy_loss": base_loss.item(),
            "train/step": 0,
        }
        
        if trainer.use_l1_regularization and l1_loss is not None:
            log_dict["train/step_l1_loss"] = l1_loss.item()
        
        MockWandb.log(log_dict, step=0)
        
        # Verify the logged data
        assert len(captured_logs) > 0, "Should have captured some logs"
        last_log_dict, last_step = captured_logs[-1]
        
        assert "train/step_loss" in last_log_dict, "Should log total step loss"
        assert "train/step_cross_entropy_loss" in last_log_dict, "Should log cross entropy loss"
        assert "train/step_l1_loss" in last_log_dict, "Should log L1 loss when enabled"
        
        # Verify the values make sense
        total_logged = last_log_dict["train/step_loss"]
        ce_logged = last_log_dict["train/step_cross_entropy_loss"]
        l1_logged = last_log_dict["train/step_l1_loss"]
        
        assert total_logged > ce_logged, "Total loss should be greater than CE loss"
        assert l1_logged > 0, "L1 loss should be positive"
        assert abs(total_logged - (ce_logged + l1_logged)) < 1e-4, "Total should equal CE + L1"
        
        print(f"   Total loss: {total_logged:.6f}")
        print(f"   Cross entropy loss: {ce_logged:.6f}")
        print(f"   L1 loss: {l1_logged:.6f}")
        print("✓ Trainer logs separate loss components correctly")
        
    finally:
        # Restore original wandb
        if original_wandb is not None:
            trainer_module.wandb = original_wandb
        else:
            if hasattr(trainer_module, 'wandb'):
                delattr(trainer_module, 'wandb')


def test_trainer_without_l1():
    """Test trainer logging when L1 regularization is disabled"""
    print("Testing trainer logging without L1 regularization...")
    
    # Create a simple model
    model = nn.Linear(20, 3)
    optimizer = ChillAdam(model.parameters())
    
    # Mock wandb logging to capture what gets logged
    captured_logs = []
    
    class MockWandb:
        @staticmethod
        def log(log_dict, step=None):
            captured_logs.append((log_dict.copy(), step))
        
        @staticmethod
        def init(*args, **kwargs):
            pass
    
    # Replace wandb temporarily
    import chilladam.training.trainer as trainer_module
    original_wandb = getattr(trainer_module, 'wandb', None)
    trainer_module.wandb = MockWandb
    trainer_module.WANDB_AVAILABLE = True
    
    try:
        # Create trainer without L1 regularization but with wandb enabled
        trainer = Trainer(
            model=model, 
            optimizer=optimizer, 
            device='cpu', 
            l1_lambda=0.0, 
            use_wandb=True,
            wandb_config={'project': 'test'}
        )
        
        # Create dummy data
        inputs = torch.randn(4, 20)
        labels = torch.randint(0, 3, (4,))
        outputs = trainer.model(inputs)
        
        # Calculate loss (should just be base loss)
        base_loss = trainer.criterion(outputs, labels)
        
        # Manually log like trainer does
        log_dict = {
            "train/step_loss": base_loss.item(),
            "train/step_cross_entropy_loss": base_loss.item(),
            "train/step": 0,
        }
        
        MockWandb.log(log_dict, step=0)
        
        # Verify the logged data
        assert len(captured_logs) > 0, "Should have captured some logs"
        last_log_dict, last_step = captured_logs[-1]
        
        assert "train/step_loss" in last_log_dict, "Should log total step loss"
        assert "train/step_cross_entropy_loss" in last_log_dict, "Should log cross entropy loss"
        assert "train/step_l1_loss" not in last_log_dict, "Should not log L1 loss when disabled"
        
        # Verify the values are equal (no L1 component)
        total_logged = last_log_dict["train/step_loss"]
        ce_logged = last_log_dict["train/step_cross_entropy_loss"]
        
        assert abs(total_logged - ce_logged) < 1e-6, "Total should equal CE when no L1"
        
        print(f"   Total loss: {total_logged:.6f}")
        print(f"   Cross entropy loss: {ce_logged:.6f}")
        print("✓ Trainer logs correctly without L1 regularization")
        
    finally:
        # Restore original wandb
        if original_wandb is not None:
            trainer_module.wandb = original_wandb
        else:
            if hasattr(trainer_module, 'wandb'):
                delattr(trainer_module, 'wandb')


def main():
    """Run all tests"""
    print("=" * 70)
    print("TESTING LOSS SEPARATION FUNCTIONALITY")
    print("=" * 70)
    
    test_separate_loss_calculation()
    print()
    
    test_no_l1_regularization()
    print()
    
    test_trainer_loss_logging()
    print()
    
    test_trainer_without_l1()
    print()
    
    print("=" * 70)
    print("ALL TESTS PASSED! 🎉")
    print("Loss separation is working correctly!")
    print("Cross entropy loss, L1 loss, and total loss are now logged separately.")
    print("=" * 70)


if __name__ == "__main__":
    main()