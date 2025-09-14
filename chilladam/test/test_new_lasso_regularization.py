#!/usr/bin/env python3
"""
Test script for the new L1 regularization functionality using loss functions.

This script tests that L1 regularization works correctly when implemented as part of
the loss function rather than in the optimizer.
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
from loss import add_l1_regularization, L1RegularizedLoss


def test_l1_regularization_basic():
    """Test that L1 regularization basic functionality works with new approach"""
    print("Testing L1 regularization basic functionality...")
    
    # Create a simple model
    model = nn.Linear(10, 1)
    
    # Test ChillAdam with new L1 approach
    optimizer_adam = ChillAdam(model.parameters())
    criterion_adam = L1RegularizedLoss(nn.MSELoss(), l1_lambda=0.01)
    assert hasattr(criterion_adam, 'l1_lambda'), "L1RegularizedLoss should have l1_lambda attribute"
    assert criterion_adam.get_l1_lambda() == 0.01, "L1 lambda should be set correctly"
    
    # Test ChillSGD with new L1 approach
    optimizer_sgd = ChillSGD(model.parameters())
    criterion_sgd = L1RegularizedLoss(nn.MSELoss(), l1_lambda=0.05)
    assert hasattr(criterion_sgd, 'l1_lambda'), "L1RegularizedLoss should have l1_lambda attribute"
    assert criterion_sgd.get_l1_lambda() == 0.05, "L1 lambda should be set correctly"
    
    print("✓ L1 regularization parameters set correctly in loss functions")


def test_l1_promotes_sparsity():
    """Test that L1 regularization promotes sparsity in weights using new approach"""
    print("Testing that L1 regularization promotes sparsity...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create two identical models
    model_no_l1 = nn.Linear(50, 1)
    model_with_l1 = nn.Linear(50, 1)
    
    # Make sure they start with identical weights
    with torch.no_grad():
        model_with_l1.weight.copy_(model_no_l1.weight)
        model_with_l1.bias.copy_(model_no_l1.bias)
    
    # Create optimizers (no L1 in optimizers anymore)
    optimizer_no_l1 = ChillAdam(model_no_l1.parameters())
    optimizer_with_l1 = ChillAdam(model_with_l1.parameters())
    
    # Create loss functions
    criterion_no_l1 = nn.MSELoss()
    criterion_with_l1 = L1RegularizedLoss(nn.MSELoss(), l1_lambda=0.1)
    
    # Create some dummy data
    x = torch.randn(100, 50)
    y = torch.randn(100, 1)
    
    # Train for a few steps
    for _ in range(20):
        # No L1 regularization
        optimizer_no_l1.zero_grad()
        output_no_l1 = model_no_l1(x)
        loss_no_l1 = criterion_no_l1(output_no_l1, y)
        loss_no_l1.backward()
        optimizer_no_l1.step()
        
        # With L1 regularization
        optimizer_with_l1.zero_grad()
        output_with_l1 = model_with_l1(x)
        loss_with_l1 = criterion_with_l1(output_with_l1, y, model_with_l1)
        loss_with_l1.backward()
        optimizer_with_l1.step()
    
    # Count parameters close to zero (sparsity measure)
    tolerance = 1e-3
    weights_no_l1 = model_no_l1.weight.data.abs()
    weights_with_l1 = model_with_l1.weight.data.abs()
    
    sparse_count_no_l1 = (weights_no_l1 < tolerance).sum().item()
    sparse_count_with_l1 = (weights_with_l1 < tolerance).sum().item()
    
    print(f"Weights close to zero (no L1): {sparse_count_no_l1}")
    print(f"Weights close to zero (with L1): {sparse_count_with_l1}")
    
    # L1 regularization should promote more sparsity
    assert sparse_count_with_l1 >= sparse_count_no_l1, "L1 regularization should promote sparsity"
    
    print("✓ L1 regularization promotes sparsity as expected")


def test_chillsgd_l1_regularization():
    """Test L1 regularization with ChillSGD optimizer using new approach"""
    print("Testing ChillSGD with L1 regularization via loss function...")
    
    # Create a simple model
    model = nn.Linear(10, 1)
    optimizer = ChillSGD(model.parameters())
    criterion = L1RegularizedLoss(nn.MSELoss(), l1_lambda=0.01)
    
    # Create some dummy data
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    # Forward pass
    output = model(x)
    loss = criterion(output, y, model)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients exist
    for p in model.parameters():
        assert p.grad is not None, "Gradients should exist"
    
    # Optimization step (should not raise errors)
    optimizer.step()
    
    print("✓ ChillSGD with L1 regularization via loss function works correctly")


def test_functional_approach():
    """Test the functional add_l1_regularization approach"""
    print("Testing functional L1 regularization approach...")
    
    model = nn.Linear(10, 1)
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    # Compute base loss
    output = model(x)
    base_loss = nn.MSELoss()(output, y)
    
    # Test with zero L1 (should return same loss)
    no_l1_loss = add_l1_regularization(base_loss, model, l1_lambda=0.0)
    assert torch.equal(base_loss, no_l1_loss), "Zero L1 should return original loss"
    
    # Test with positive L1 (should be larger)
    with_l1_loss = add_l1_regularization(base_loss, model, l1_lambda=0.01)
    assert with_l1_loss > base_loss, "L1 regularization should increase loss"
    
    print("✓ Functional L1 regularization approach works correctly")


def test_factory_compatibility():
    """Test that factory method works without L1 parameters"""
    print("Testing factory compatibility...")
    
    model = nn.Linear(10, 1)
    
    # Test ChillAdam via factory (no L1 lambda should work fine)
    optimizer1 = create_optimizer('chilladam', model.parameters())
    assert optimizer1 is not None, "Factory should create ChillAdam without L1"
    
    # Test ChillSGD via factory (no L1 lambda should work fine)
    optimizer2 = create_optimizer('chillsgd', model.parameters())
    assert optimizer2 is not None, "Factory should create ChillSGD without L1"
    
    print("✓ Factory method works correctly without L1 parameters")


def test_error_handling():
    """Test error handling for invalid L1 lambda values"""
    print("Testing error handling for invalid L1 lambda values...")
    
    # Test negative L1 lambda in L1RegularizedLoss (should raise error)
    try:
        L1RegularizedLoss(nn.MSELoss(), l1_lambda=-0.1)
        assert False, "Should raise error for negative L1 lambda"
    except ValueError as e:
        assert "l1_lambda must be non-negative" in str(e), "Should mention non-negative l1_lambda in error"
    
    # Test set_l1_lambda with negative value
    criterion = L1RegularizedLoss(nn.MSELoss(), l1_lambda=0.0)
    try:
        criterion.set_l1_lambda(-0.1)
        assert False, "Should raise error for negative L1 lambda in setter"
    except ValueError as e:
        assert "l1_lambda must be non-negative" in str(e), "Should mention non-negative l1_lambda in error"
    
    print("✓ Error handling works correctly for invalid L1 lambda values")


def test_equivalence_with_old_approach():
    """Test that new approach produces similar results to what old approach would"""
    print("Testing equivalence with manual L1 implementation...")
    
    # Set random seed for reproducibility
    torch.manual_seed(123)
    
    model = nn.Linear(20, 1)
    x = torch.randn(10, 20)
    y = torch.randn(10, 1)
    
    # New approach
    criterion_new = L1RegularizedLoss(nn.MSELoss(), l1_lambda=0.01)
    output = model(x)
    loss_new = criterion_new(output, y, model)
    
    # Manual approach (what the old optimizer was doing conceptually)
    base_loss = nn.MSELoss()(output, y)
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    loss_manual = base_loss + 0.01 * l1_norm
    
    # They should be the same
    assert torch.allclose(loss_new, loss_manual, atol=1e-6), "New approach should match manual calculation"
    
    print("✓ New approach produces equivalent results to manual L1 calculation")


def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING NEW L1 REGULARIZATION LOSS-BASED FUNCTIONALITY")
    print("=" * 60)
    
    test_l1_regularization_basic()
    print()
    
    test_l1_promotes_sparsity()
    print()
    
    test_chillsgd_l1_regularization()
    print()
    
    test_functional_approach()
    print()
    
    test_factory_compatibility()
    print()
    
    test_error_handling()
    print()
    
    test_equivalence_with_old_approach()
    print()
    
    print("=" * 60)
    print("ALL TESTS PASSED! 🎉")
    print("L1 regularization successfully moved from optimizers to loss functions")
    print("The new approach provides better separation of concerns and follows")
    print("standard PyTorch patterns where regularization is handled in the loss.")
    print("=" * 60)


if __name__ == "__main__":
    main()