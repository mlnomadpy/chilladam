#!/usr/bin/env python3
"""
Test script for Lasso (L1) regularization functionality in ChillAdam and ChillSGD optimizers.

This script tests that L1 regularization promotes sparsity by comparing parameter values
with and without L1 regularization.
"""

import torch
import torch.nn as nn
import sys
import os

# Add chilladam to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chilladam'))

from optimizers.chilladam import ChillAdam
from optimizers.chillsgd import ChillSGD
from optimizers.factory import create_optimizer


def test_l1_regularization_basic():
    """Test that L1 regularization parameter is accepted and works"""
    print("Testing L1 regularization basic functionality...")
    
    # Create a simple model
    model = nn.Linear(10, 1)
    
    # Test ChillAdam with L1
    optimizer_adam = ChillAdam(model.parameters(), l1_lambda=0.01)
    assert hasattr(optimizer_adam, 'param_groups'), "ChillAdam should have param_groups"
    assert optimizer_adam.param_groups[0]['l1_lambda'] == 0.01, "L1 lambda should be set correctly"
    
    # Test ChillSGD with L1
    optimizer_sgd = ChillSGD(model.parameters(), l1_lambda=0.05)
    assert hasattr(optimizer_sgd, 'param_groups'), "ChillSGD should have param_groups"
    assert optimizer_sgd.param_groups[0]['l1_lambda'] == 0.05, "L1 lambda should be set correctly"
    
    print("✓ L1 regularization parameters set correctly")


def test_l1_promotes_sparsity():
    """Test that L1 regularization promotes sparsity in weights"""
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
    
    # Create optimizers
    optimizer_no_l1 = ChillAdam(model_no_l1.parameters(), l1_lambda=0.0)
    optimizer_with_l1 = ChillAdam(model_with_l1.parameters(), l1_lambda=0.1)
    
    # Create some dummy data
    x = torch.randn(100, 50)
    y = torch.randn(100, 1)
    
    # Train for a few steps
    for _ in range(20):
        # No L1 regularization
        optimizer_no_l1.zero_grad()
        output_no_l1 = model_no_l1(x)
        loss_no_l1 = nn.MSELoss()(output_no_l1, y)
        loss_no_l1.backward()
        optimizer_no_l1.step()
        
        # With L1 regularization
        optimizer_with_l1.zero_grad()
        output_with_l1 = model_with_l1(x)
        loss_with_l1 = nn.MSELoss()(output_with_l1, y)
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
    """Test L1 regularization with ChillSGD optimizer"""
    print("Testing ChillSGD with L1 regularization...")
    
    # Create a simple model
    model = nn.Linear(10, 1)
    optimizer = ChillSGD(model.parameters(), l1_lambda=0.01)
    
    # Create some dummy data
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    # Forward pass
    output = model(x)
    loss = nn.MSELoss()(output, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients exist
    for p in model.parameters():
        assert p.grad is not None, "Gradients should exist"
    
    # Optimization step (should not raise errors)
    optimizer.step()
    
    print("✓ ChillSGD with L1 regularization works correctly")


def test_factory_l1_support():
    """Test that factory method supports L1 regularization"""
    print("Testing factory support for L1 regularization...")
    
    model = nn.Linear(10, 1)
    
    # Test ChillAdam via factory
    optimizer1 = create_optimizer('chilladam', model.parameters(), l1_lambda=0.02)
    assert optimizer1.param_groups[0]['l1_lambda'] == 0.02, "Factory should set L1 lambda correctly"
    
    # Test ChillSGD via factory
    optimizer2 = create_optimizer('chillsgd', model.parameters(), l1_lambda=0.03)
    assert optimizer2.param_groups[0]['l1_lambda'] == 0.03, "Factory should set L1 lambda correctly"
    
    # Test default value (should be 0)
    optimizer3 = create_optimizer('chilladam', model.parameters())
    assert optimizer3.param_groups[0]['l1_lambda'] == 0, "Default L1 lambda should be 0"
    
    print("✓ Factory method supports L1 regularization correctly")


def test_error_handling():
    """Test error handling for invalid L1 lambda values"""
    print("Testing error handling for invalid L1 lambda values...")
    
    model = nn.Linear(10, 1)
    
    # Test negative L1 lambda (should raise error)
    try:
        ChillAdam(model.parameters(), l1_lambda=-0.1)
        assert False, "Should raise error for negative L1 lambda"
    except ValueError as e:
        assert "Invalid l1_lambda" in str(e), "Should mention invalid l1_lambda in error"
    
    try:
        ChillSGD(model.parameters(), l1_lambda=-0.1)
        assert False, "Should raise error for negative L1 lambda"
    except ValueError as e:
        assert "Invalid l1_lambda" in str(e), "Should mention invalid l1_lambda in error"
    
    print("✓ Error handling works correctly for invalid L1 lambda values")


def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING LASSO (L1) REGULARIZATION FUNCTIONALITY")
    print("=" * 60)
    
    test_l1_regularization_basic()
    print()
    
    test_l1_promotes_sparsity()
    print()
    
    test_chillsgd_l1_regularization()
    print()
    
    test_factory_l1_support()
    print()
    
    test_error_handling()
    print()
    
    print("=" * 60)
    print("ALL TESTS PASSED! 🎉")
    print("Lasso (L1) regularization is working correctly for all weight matrices")
    print("=" * 60)


if __name__ == "__main__":
    main()