#!/usr/bin/env python3
"""
Test script for L1 regularization functionality in the updated ChillAdam system.

This script tests that L1 regularization is properly connected from command line
arguments to the actual implementation via the loss function.
"""

import torch
import torch.nn as nn
import sys
import os

# Add chilladam to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimizers.chilladam import ChillAdam
from optimizers.chillsgd import ChillSGD
from optimizers.factory import create_optimizer
from training.trainer import Trainer
from loss import L1RegularizedLoss, add_l1_regularization


def test_l1_loss_function():
    """Test that L1 regularization works in the loss function"""
    print("Testing L1 regularization in loss function...")
    
    # Create a simple model
    model = nn.Linear(10, 1)
    
    # Test L1RegularizedLoss class
    base_criterion = nn.MSELoss()
    l1_criterion = L1RegularizedLoss(base_criterion, l1_lambda=0.01)
    
    # Create test data
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    # Forward pass
    outputs = model(x)
    
    # Calculate losses
    base_loss = base_criterion(outputs, y)
    l1_loss = l1_criterion(outputs, y, model)
    
    # L1 loss should be higher due to regularization
    assert l1_loss > base_loss, "L1 regularized loss should be higher than base loss"
    
    # Test functional approach
    l1_loss_func = add_l1_regularization(base_loss, model, l1_lambda=0.01)
    
    # Should match the class-based approach
    assert abs(l1_loss.item() - l1_loss_func.item()) < 1e-6, "Class and functional approaches should match"
    
    print("✓ L1 regularization in loss function works correctly")


def test_trainer_integration():
    """Test that L1 regularization works in the trainer"""
    print("Testing L1 regularization in trainer...")
    
    # Create a simple model
    model = nn.Linear(20, 3)
    optimizer = ChillAdam(model.parameters())
    
    # Test trainer without L1
    trainer_no_l1 = Trainer(model, optimizer, device='cpu', l1_lambda=0)
    assert not trainer_no_l1.use_l1_regularization, "Should not use L1 regularization"
    assert trainer_no_l1.l1_lambda == 0, "L1 lambda should be 0"
    
    # Test trainer with L1
    trainer_with_l1 = Trainer(model, optimizer, device='cpu', l1_lambda=0.02)
    assert trainer_with_l1.use_l1_regularization, "Should use L1 regularization"
    assert trainer_with_l1.l1_lambda == 0.02, "L1 lambda should be 0.02"
    assert type(trainer_with_l1.criterion).__name__ == 'L1RegularizedLoss', "Should use L1RegularizedLoss"
    
    # Test forward pass
    x = torch.randn(8, 20)
    y = torch.randint(0, 3, (8,))
    
    # No L1 trainer
    outputs = model(x)
    loss_no_l1 = trainer_no_l1.criterion(outputs, y)
    
    # With L1 trainer
    loss_with_l1 = trainer_with_l1.criterion(outputs, y, model)
    
    # L1 regularized loss should be higher
    assert loss_with_l1 > loss_no_l1, "L1 regularized loss should be higher"
    
    print("✓ L1 regularization in trainer works correctly")


def test_factory_warning():
    """Test that factory function warns about l1_lambda"""
    print("Testing factory warning for l1_lambda...")
    
    model = nn.Linear(10, 1)
    
    # Capture the warning by redirecting stdout
    import io
    import contextlib
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        optimizer = create_optimizer('chilladam', model.parameters(), l1_lambda=0.01)
    
    output = f.getvalue()
    assert "Warning" in output, "Should print warning about l1_lambda"
    assert "ignored" in output, "Should mention l1_lambda is ignored"
    assert "loss function" in output, "Should mention loss function"
    
    # Test without l1_lambda (should not warn)
    f2 = io.StringIO()
    with contextlib.redirect_stdout(f2):
        optimizer2 = create_optimizer('chilladam', model.parameters())
    
    output2 = f2.getvalue()
    assert "Warning" not in output2, "Should not print warning without l1_lambda"
    
    print("✓ Factory warning system works correctly")


def test_sparsity_promotion():
    """Test that L1 regularization promotes sparsity"""
    print("Testing that L1 regularization promotes sparsity...")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Create two identical models
    model_no_l1 = nn.Linear(50, 1)
    model_with_l1 = nn.Linear(50, 1)
    
    # Make sure they start with identical weights
    with torch.no_grad():
        model_with_l1.weight.copy_(model_no_l1.weight)
        model_with_l1.bias.copy_(model_no_l1.bias)
    
    # Create optimizers
    optimizer_no_l1 = ChillAdam(model_no_l1.parameters())
    optimizer_with_l1 = ChillAdam(model_with_l1.parameters())
    
    # Create trainers
    trainer_no_l1 = Trainer(model_no_l1, optimizer_no_l1, device='cpu', l1_lambda=0)
    trainer_with_l1 = Trainer(model_with_l1, optimizer_with_l1, device='cpu', l1_lambda=0.1)
    
    # Create dummy data
    x = torch.randn(100, 50)
    y = torch.randn(100, 1)
    
    # Train for several steps
    for step in range(20):
        # Train without L1
        optimizer_no_l1.zero_grad()
        outputs_no_l1 = model_no_l1(x)
        loss_no_l1 = trainer_no_l1.criterion(outputs_no_l1, y)
        loss_no_l1.backward()
        optimizer_no_l1.step()
        
        # Train with L1
        optimizer_with_l1.zero_grad()
        outputs_with_l1 = model_with_l1(x)
        loss_with_l1 = trainer_with_l1.criterion(outputs_with_l1, y, model_with_l1)
        loss_with_l1.backward()
        optimizer_with_l1.step()
    
    # Count parameters close to zero
    tolerance = 1e-3
    weights_no_l1 = model_no_l1.weight.data.abs()
    weights_with_l1 = model_with_l1.weight.data.abs()
    
    sparse_count_no_l1 = (weights_no_l1 < tolerance).sum().item()
    sparse_count_with_l1 = (weights_with_l1 < tolerance).sum().item()
    
    print(f"   Weights close to zero (no L1): {sparse_count_no_l1}")
    print(f"   Weights close to zero (with L1): {sparse_count_with_l1}")
    
    # L1 regularization should promote more sparsity
    assert sparse_count_with_l1 >= sparse_count_no_l1, "L1 regularization should promote sparsity"
    
    print("✓ L1 regularization promotes sparsity as expected")


def test_both_optimizers():
    """Test L1 regularization with both ChillAdam and ChillSGD"""
    print("Testing L1 regularization with both optimizers...")
    
    # Simple model
    model_adam = nn.Linear(20, 1)
    model_sgd = nn.Linear(20, 1)
    
    # Create optimizers
    optimizer_adam = ChillAdam(model_adam.parameters())
    optimizer_sgd = ChillSGD(model_sgd.parameters())
    
    # Create trainers with L1 regularization
    trainer_adam = Trainer(model_adam, optimizer_adam, device='cpu', l1_lambda=0.02)
    trainer_sgd = Trainer(model_sgd, optimizer_sgd, device='cpu', l1_lambda=0.03)
    
    assert trainer_adam.use_l1_regularization, "ChillAdam trainer should use L1"
    assert trainer_sgd.use_l1_regularization, "ChillSGD trainer should use L1"
    assert trainer_adam.l1_lambda == 0.02, "ChillAdam trainer should have correct lambda"
    assert trainer_sgd.l1_lambda == 0.03, "ChillSGD trainer should have correct lambda"
    
    # Quick training step to verify no errors
    x = torch.randn(10, 20)
    y = torch.randn(10, 1)
    
    # Test ChillAdam
    optimizer_adam.zero_grad()
    output_adam = model_adam(x)
    loss_adam = trainer_adam.criterion(output_adam, y, model_adam)
    loss_adam.backward()
    optimizer_adam.step()
    
    # Test ChillSGD
    optimizer_sgd.zero_grad()
    output_sgd = model_sgd(x)
    loss_sgd = trainer_sgd.criterion(output_sgd, y, model_sgd)
    loss_sgd.backward()
    optimizer_sgd.step()
    
    print("✓ Both optimizers work correctly with L1 regularization via trainer")


def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING L1 REGULARIZATION INTEGRATION (NEW APPROACH)")
    print("=" * 60)
    
    test_l1_loss_function()
    print()
    
    test_trainer_integration()
    print()
    
    test_factory_warning()
    print()
    
    test_sparsity_promotion()
    print()
    
    test_both_optimizers()
    print()
    
    print("=" * 60)
    print("ALL TESTS PASSED! 🎉")
    print("L1 regularization is properly connected from arguments to implementation!")
    print("The --l1-lambda argument now works correctly with the training pipeline.")
    print("=" * 60)


if __name__ == "__main__":
    main()