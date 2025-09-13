#!/usr/bin/env python3
"""
Integration test for ChillSGD with a simple training scenario.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add chilladam to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_chillsgd_training():
    """Test ChillSGD in a simple training loop"""
    print("Testing ChillSGD in a training scenario...")
    
    from chilladam.optimizers import ChillSGD
    
    # Create a simple model
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(4, 10)
            self.fc2 = nn.Linear(10, 2)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleNet()
    optimizer = ChillSGD(model.parameters(), min_lr=1e-5, max_lr=0.1, eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    
    # Create synthetic dataset
    torch.manual_seed(42)
    X = torch.randn(100, 4)
    y = torch.randint(0, 2, (100,))
    
    # Training loop for several iterations
    model.train()
    initial_loss = None
    final_loss = None
    
    for epoch in range(20):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        if initial_loss is None:
            initial_loss = loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch == 19:
            final_loss = loss.item()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}: Loss = {loss.item():.6f}")
    
    # Check that loss decreased
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss:   {final_loss:.6f}")
    print(f"  Loss reduction: {initial_loss - final_loss:.6f}")
    
    assert final_loss < initial_loss, f"Loss should decrease during training. Initial: {initial_loss}, Final: {final_loss}"
    
    # Test inference
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 4)
        output = model(test_input)
        prediction = torch.argmax(output, dim=1)
        print(f"  Sample prediction: {prediction.item()}")
    
    print("✓ ChillSGD training loop working correctly")

def test_chillsgd_vs_chilladam_performance():
    """Compare ChillSGD vs ChillAdam on the same task"""
    print("\nComparing ChillSGD vs ChillAdam performance...")
    
    from chilladam.optimizers import ChillSGD, ChillAdam
    
    # Create identical models
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(4, 10)
            self.fc2 = nn.Linear(10, 2)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    torch.manual_seed(42)
    model_sgd = SimpleNet()
    torch.manual_seed(42)
    model_adam = SimpleNet()
    
    # Ensure they start with the same parameters
    for p1, p2 in zip(model_sgd.parameters(), model_adam.parameters()):
        assert torch.allclose(p1, p2), "Models should start with identical parameters"
    
    optimizer_sgd = ChillSGD(model_sgd.parameters(), min_lr=1e-5, max_lr=0.1)
    optimizer_adam = ChillAdam(model_adam.parameters(), min_lr=1e-5, max_lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Create synthetic dataset
    torch.manual_seed(123)
    X = torch.randn(100, 4)
    y = torch.randint(0, 2, (100,))
    
    # Train both models
    sgd_losses = []
    adam_losses = []
    
    for epoch in range(10):
        # ChillSGD
        model_sgd.train()
        outputs_sgd = model_sgd(X)
        loss_sgd = criterion(outputs_sgd, y)
        optimizer_sgd.zero_grad()
        loss_sgd.backward()
        optimizer_sgd.step()
        sgd_losses.append(loss_sgd.item())
        
        # ChillAdam  
        model_adam.train()
        outputs_adam = model_adam(X)
        loss_adam = criterion(outputs_adam, y)
        optimizer_adam.zero_grad()
        loss_adam.backward()
        optimizer_adam.step()
        adam_losses.append(loss_adam.item())
    
    print(f"  ChillSGD final loss:  {sgd_losses[-1]:.6f}")
    print(f"  ChillAdam final loss: {adam_losses[-1]:.6f}")
    
    # Both should reduce loss, but may converge differently
    assert sgd_losses[-1] < sgd_losses[0], "ChillSGD should reduce loss"
    assert adam_losses[-1] < adam_losses[0], "ChillAdam should reduce loss"
    
    print("✓ Both ChillSGD and ChillAdam successfully reduce loss")

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("CHILLSGD INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        test_chillsgd_training()
        test_chillsgd_vs_chilladam_performance()
        
        print("\n" + "=" * 60)
        print("ALL CHILLSGD INTEGRATION TESTS PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)