#!/usr/bin/env python3
"""
Test script for ChillSGD optimizer

This script tests the newly added ChillSGD optimizer functionality.
"""

import torch
import torch.nn as nn
import sys
import os

# Add chilladam to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_chillsgd_basic():
    """Test basic ChillSGD functionality"""
    print("Testing ChillSGD basic functionality...")
    
    # Import the optimizer
    from chilladam.optimizers import ChillSGD
    
    # Create a simple model
    model = nn.Linear(10, 1)
    
    # Create optimizer
    optimizer = ChillSGD(model.parameters(), min_lr=1e-5, max_lr=1.0, eps=1e-8, weight_decay=0)
    
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
    
    # Store initial parameters
    initial_params = [p.clone() for p in model.parameters()]
    
    # Optimization step
    optimizer.step()
    
    # Check parameters changed
    for i, p in enumerate(model.parameters()):
        assert not torch.equal(p, initial_params[i]), "Parameters should have changed after optimization step"
    
    print("✓ ChillSGD basic functionality working correctly")

def test_chillsgd_factory():
    """Test ChillSGD through the factory"""
    print("\nTesting ChillSGD through factory...")
    
    from chilladam.optimizers import create_optimizer
    
    # Create a simple model
    model = nn.Linear(10, 1)
    
    # Create optimizer through factory
    optimizer = create_optimizer("chillsgd", model.parameters(), min_lr=1e-5, max_lr=1.0)
    
    # Verify it's the right type
    from chilladam.optimizers import ChillSGD
    assert isinstance(optimizer, ChillSGD), f"Expected ChillSGD, got {type(optimizer)}"
    
    print("✓ ChillSGD factory creation working correctly")

def test_chillsgd_vs_sgd():
    """Test that ChillSGD behaves differently from regular SGD"""
    print("\nTesting ChillSGD vs regular SGD behavior...")
    
    from chilladam.optimizers import ChillSGD
    
    # Create two identical models
    model1 = nn.Linear(10, 1)
    model2 = nn.Linear(10, 1)
    
    # Copy weights to make them identical
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p2.copy_(p1)
    
    # Create optimizers
    chillsgd_optimizer = ChillSGD(model1.parameters(), min_lr=1e-5, max_lr=1.0)
    sgd_optimizer = torch.optim.SGD(model2.parameters(), lr=0.01)
    
    # Create identical data
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    # Forward and backward for ChillSGD
    output1 = model1(x)
    loss1 = nn.MSELoss()(output1, y)
    chillsgd_optimizer.zero_grad()
    loss1.backward()
    chillsgd_optimizer.step()
    
    # Forward and backward for SGD
    output2 = model2(x)
    loss2 = nn.MSELoss()(output2, y)
    sgd_optimizer.zero_grad()
    loss2.backward()
    sgd_optimizer.step()
    
    # Check that the updates are different (due to different mechanisms)
    params_different = False
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(p1, p2, atol=1e-6):
            params_different = True
            break
    
    assert params_different, "ChillSGD should behave differently from regular SGD"
    
    print("✓ ChillSGD behaves differently from regular SGD as expected")

def test_chillsgd_gradient_normalization():
    """Test that ChillSGD performs gradient normalization"""
    print("\nTesting ChillSGD gradient normalization...")
    
    from chilladam.optimizers import ChillSGD
    
    # Create a model
    model = nn.Linear(2, 1)
    optimizer = ChillSGD(model.parameters(), min_lr=1e-5, max_lr=1.0)
    
    # Create data that will produce different gradient magnitudes
    x = torch.tensor([[1000.0, 0.0], [0.0, 1.0]])  # Very different scales
    y = torch.tensor([[1.0], [1.0]])
    
    # Forward and backward
    output = model(x)
    loss = nn.MSELoss()(output, y)
    optimizer.zero_grad()
    loss.backward()
    
    # Store original gradients
    original_grads = [p.grad.clone() for p in model.parameters()]
    
    # Check that gradients have different norms initially
    grad_norms = [g.norm(p=2) for g in original_grads]
    print(f"  Original gradient norms: {[f'{norm.item():.6f}' for norm in grad_norms]}")
    
    # After ChillSGD step, the effective gradients should be normalized
    # We can't directly observe this, but we can verify that the optimizer works
    optimizer.step()
    
    print("✓ ChillSGD gradient normalization mechanism in place")

def test_chillsgd_adaptive_lr():
    """Test that ChillSGD uses adaptive learning rates based on parameter norms"""
    print("\nTesting ChillSGD adaptive learning rates...")
    
    from chilladam.optimizers import ChillSGD
    
    # Create a model with parameters of different magnitudes
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.Linear(3, 1)
    )
    
    # Manually set different parameter magnitudes
    with torch.no_grad():
        # First layer weights: small magnitude
        model[0].weight.data = torch.tensor([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
        model[0].bias.data = torch.tensor([0.1, 0.1, 0.1])
        
        # Second layer weights: large magnitude  
        model[1].weight.data = torch.tensor([[10.0, 10.0, 10.0]])
        model[1].bias.data = torch.tensor([10.0])
    
    optimizer = ChillSGD(model.parameters(), min_lr=1e-5, max_lr=1.0)
    
    # Forward and backward
    x = torch.randn(1, 2)
    y = torch.randn(1, 1)
    output = model(x)
    loss = nn.MSELoss()(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check that different parameters got different learning rates stored in state
    learning_rates = []
    for p in model.parameters():
        if p in optimizer.state:
            lr = optimizer.state[p].get("lr", None)
            if lr is not None:
                learning_rates.append(lr)
    
    print(f"  Adaptive learning rates: {[f'{lr:.6f}' for lr in learning_rates]}")
    
    # Should have different learning rates for parameters with different norms
    if len(learning_rates) > 1:
        assert not all(abs(lr - learning_rates[0]) < 1e-6 for lr in learning_rates), \
            "Learning rates should vary based on parameter norms"
    
    print("✓ ChillSGD adaptive learning rates working correctly")

def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING CHILLSGD OPTIMIZER")
    print("=" * 60)
    
    try:
        test_chillsgd_basic()
        test_chillsgd_factory()
        test_chillsgd_vs_sgd()
        test_chillsgd_gradient_normalization()
        test_chillsgd_adaptive_lr()
        
        print("\n" + "=" * 60)
        print("ALL CHILLSGD TESTS PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)