#!/usr/bin/env python3
"""
Demonstration script showing Lasso (L1) regularization in action using the new loss-based approach.

This script creates a simple model and demonstrates that L1 regularization
actually promotes sparsity in the weight matrices. L1 regularization is now
applied through the loss function rather than the optimizer.
"""

import torch
import torch.nn as nn
import sys
import os

# Add chilladam to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'chilladam'))

from optimizers.chilladam import ChillAdam
from optimizers.chillsgd import ChillSGD
from loss import add_l1_regularization, L1RegularizedLoss


def demonstrate_lasso_regularization():
    """Demonstrate Lasso regularization promoting sparsity using the new loss-based approach"""
    print("=" * 60)
    print("DEMONSTRATING LASSO (L1) REGULARIZATION FOR WEIGHT SPARSITY")
    print("Using the new loss-based approach")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(123)
    
    # Create a simple multi-layer network
    model_no_lasso = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    model_with_lasso = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # Initialize both models with identical weights
    with torch.no_grad():
        for p1, p2 in zip(model_no_lasso.parameters(), model_with_lasso.parameters()):
            p2.copy_(p1)
    
    # Create optimizers (no L1 in optimizers anymore)
    optimizer_no_lasso = ChillAdam(model_no_lasso.parameters(), min_lr=1e-4, max_lr=0.01)
    optimizer_with_lasso = ChillAdam(model_with_lasso.parameters(), min_lr=1e-4, max_lr=0.01)
    
    # Create loss functions - this is the new approach!
    criterion_no_lasso = nn.MSELoss()
    criterion_with_lasso = L1RegularizedLoss(nn.MSELoss(), l1_lambda=0.01)
    
    print(f"Training without L1 regularization: using standard MSELoss")
    print(f"Training with L1 regularization: using L1RegularizedLoss with l1_lambda = 0.01")
    print()
    
    # Generate some dummy regression data
    x = torch.randn(200, 100)
    y = torch.randn(200, 1)
    
    # Train for several epochs
    num_epochs = 50
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train model without L1 regularization
        optimizer_no_lasso.zero_grad()
        output_no_lasso = model_no_lasso(x)
        loss_no_lasso = criterion_no_lasso(output_no_lasso, y)
        loss_no_lasso.backward()
        optimizer_no_lasso.step()
        
        # Train model with L1 regularization - note the model parameter is passed to the loss
        optimizer_with_lasso.zero_grad()
        output_with_lasso = model_with_lasso(x)
        loss_with_lasso = criterion_with_lasso(output_with_lasso, y, model_with_lasso)
        loss_with_lasso.backward()
        optimizer_with_lasso.step()
    
    print("Training completed!")
    print()
    
    # Analyze sparsity in the weight matrices
    def count_sparse_weights(model, threshold=1e-3):
        total_params = 0
        sparse_params = 0
        for name, param in model.named_parameters():
            if 'weight' in name:  # Only check weight matrices, not biases
                param_count = param.numel()
                sparse_count = (param.abs() < threshold).sum().item()
                total_params += param_count
                sparse_params += sparse_count
                print(f"  {name}: {sparse_count}/{param_count} weights close to zero ({100*sparse_count/param_count:.1f}%)")
        return sparse_params, total_params
    
    print("SPARSITY ANALYSIS (weights with absolute value < 1e-3):")
    print()
    
    print("Model WITHOUT L1 regularization:")
    sparse_no_lasso, total_no_lasso = count_sparse_weights(model_no_lasso)
    sparsity_no_lasso = 100 * sparse_no_lasso / total_no_lasso
    print(f"  Total sparsity: {sparse_no_lasso}/{total_no_lasso} ({sparsity_no_lasso:.1f}%)")
    print()
    
    print("Model WITH L1 regularization:")
    sparse_with_lasso, total_with_lasso = count_sparse_weights(model_with_lasso)
    sparsity_with_lasso = 100 * sparse_with_lasso / total_with_lasso
    print(f"  Total sparsity: {sparse_with_lasso}/{total_with_lasso} ({sparsity_with_lasso:.1f}%)")
    print()
    
    # Show the effect
    improvement = sparsity_with_lasso - sparsity_no_lasso
    print("RESULT:")
    if improvement > 0:
        print(f"✅ L1 regularization increased sparsity by {improvement:.1f} percentage points!")
        print("🎯 Lasso regularization successfully promotes sparse weight matrices")
    else:
        print(f"⚠️  L1 regularization effect: {improvement:.1f} percentage points")
        print("ℹ️  L1 regularization effect may vary with different hyperparameters")
    
    print()
    print("🆕 NEW APPROACH: L1 regularization is now implemented in the loss function!")
    print("📖 Example usage:")
    print("    from chilladam.loss import add_l1_regularization, L1RegularizedLoss")
    print("    # Option 1: Function")
    print("    loss = add_l1_regularization(base_loss, model, l1_lambda=0.01)")
    print("    # Option 2: Class")
    print("    criterion = L1RegularizedLoss(nn.MSELoss(), l1_lambda=0.01)")
    print("    loss = criterion(outputs, targets, model)")
    print()
    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


def test_both_optimizers():
    """Test L1 regularization with both ChillAdam and ChillSGD using the new loss-based approach"""
    print("\nTesting L1 regularization with both optimizers using the new approach...")
    
    # Simple model
    model_adam = nn.Linear(20, 1)
    model_sgd = nn.Linear(20, 1)
    
    # Create optimizers (no L1 parameters)
    optimizer_adam = ChillAdam(model_adam.parameters())
    optimizer_sgd = ChillSGD(model_sgd.parameters())
    
    # Create L1 regularized loss functions
    criterion_adam = L1RegularizedLoss(nn.MSELoss(), l1_lambda=0.02)
    criterion_sgd = L1RegularizedLoss(nn.MSELoss(), l1_lambda=0.03)
    
    print(f"✓ ChillAdam with L1RegularizedLoss (l1_lambda = {criterion_adam.get_l1_lambda()})")
    print(f"✓ ChillSGD with L1RegularizedLoss (l1_lambda = {criterion_sgd.get_l1_lambda()})")
    
    # Quick training step to verify no errors
    x = torch.randn(10, 20)
    y = torch.randn(10, 1)
    
    # Test ChillAdam
    optimizer_adam.zero_grad()
    output_adam = model_adam(x)
    loss_adam = criterion_adam(output_adam, y, model_adam)
    loss_adam.backward()
    optimizer_adam.step()
    
    # Test ChillSGD
    optimizer_sgd.zero_grad()
    output_sgd = model_sgd(x)
    loss_sgd = criterion_sgd(output_sgd, y, model_sgd)
    loss_sgd.backward()
    optimizer_sgd.step()
    
    print("✓ Both optimizers work correctly with L1 regularization via loss function")
    
    # Demonstrate the functional approach too
    print("\nDemonstrating the functional approach:")
    base_loss = nn.MSELoss()(model_adam(x), y)
    l1_loss = add_l1_regularization(base_loss, model_adam, l1_lambda=0.02)
    print(f"✓ Functional approach: base_loss={base_loss.item():.4f}, l1_loss={l1_loss.item():.4f}")
    
    print("\n🔧 NEW API: L1 regularization is now controlled by loss functions, not optimizers!")
    print("   This provides better separation of concerns and follows PyTorch conventions.")


if __name__ == "__main__":
    demonstrate_lasso_regularization()
    test_both_optimizers()
    print("\n🎉 All demonstrations completed successfully!")
    print("Lasso (L1) regularization is now implemented in the loss function for better modularity!")