"""
Test scheduler functionality.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add the chilladam package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from chilladam.schedulers import create_scheduler, get_scheduler_info
from chilladam.optimizers import create_optimizer


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


def test_scheduler_factory():
    """Test scheduler factory creation."""
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Test cosine scheduler
    scheduler = create_scheduler("cosine", optimizer, t_max=10, eta_min=1e-6)
    assert scheduler is not None
    assert hasattr(scheduler, 'step')
    
    # Test no scheduler
    scheduler = create_scheduler("none", optimizer)
    assert scheduler is None


def test_cosine_scheduler_functionality():
    """Test that cosine scheduler actually changes learning rate."""
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = create_scheduler("cosine", optimizer, t_max=10, eta_min=1e-6)
    
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Step the scheduler a few times
    for _ in range(5):
        scheduler.step()
    
    # Learning rate should have changed
    new_lr = optimizer.param_groups[0]['lr']
    assert new_lr != initial_lr
    assert new_lr > 1e-6  # Should be above eta_min


def test_scheduler_with_chilladam():
    """Test scheduler integration with ChillAdam optimizer."""
    model = SimpleModel()
    
    # Create ChillAdam optimizer
    optimizer = create_optimizer(
        "chilladam", 
        model.parameters(),
        min_lr=1e-5,
        max_lr=1.0
    )
    
    # Create cosine scheduler
    scheduler = create_scheduler("cosine", optimizer, t_max=10, eta_min=1e-6)
    
    # Should work without errors
    assert scheduler is not None
    scheduler.step()


def test_scheduler_info():
    """Test scheduler info function."""
    info = get_scheduler_info()
    assert isinstance(info, dict)
    assert "cosine" in info
    assert "step" in info
    assert "none" in info


def test_unsupported_scheduler():
    """Test that unsupported scheduler raises ValueError."""
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    with pytest.raises(ValueError):
        create_scheduler("unsupported_scheduler", optimizer)


def test_step_scheduler():
    """Test step scheduler creation and functionality."""
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = create_scheduler("step", optimizer, step_size=5, gamma=0.1)
    
    assert scheduler is not None
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Step 4 times (before step_size)
    for _ in range(4):
        scheduler.step()
    
    # LR should be the same
    assert optimizer.param_groups[0]['lr'] == initial_lr
    
    # Step once more (reach step_size)
    scheduler.step()
    
    # LR should have decreased
    assert optimizer.param_groups[0]['lr'] == initial_lr * 0.1


def test_exponential_scheduler():
    """Test exponential scheduler creation and functionality."""
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = create_scheduler("exponential", optimizer, gamma=0.95)
    
    assert scheduler is not None
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Step once
    scheduler.step()
    
    # LR should have decreased
    assert optimizer.param_groups[0]['lr'] == initial_lr * 0.95


def test_cosine_warmup_scheduler():
    """Test cosine warmup scheduler creation and functionality."""
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = create_scheduler(
        "cosine_warmup", 
        optimizer, 
        total_epochs=10, 
        warmup_epochs=3, 
        eta_min=1e-6
    )
    
    assert scheduler is not None
    
    # Test initial state (should be 0 with warmup)
    assert optimizer.param_groups[0]['lr'] == 0.0
    
    # Test warmup phase
    scheduler.step()  # epoch 0
    lr_epoch_0 = optimizer.param_groups[0]['lr']
    assert lr_epoch_0 > 0 and lr_epoch_0 < 0.1
    
    scheduler.step()  # epoch 1 
    lr_epoch_1 = optimizer.param_groups[0]['lr']
    assert lr_epoch_1 > lr_epoch_0
    
    scheduler.step()  # epoch 2 (end of warmup)
    lr_epoch_2 = optimizer.param_groups[0]['lr']
    assert lr_epoch_1 < lr_epoch_2  # Should increase during warmup
    assert lr_epoch_2 <= 0.1  # Should not exceed base LR


def test_cosine_warmup_scheduler_info():
    """Test that cosine_warmup scheduler appears in info."""
    info = get_scheduler_info()
    assert "cosine_warmup" in info
    assert "Linear Warmup" in info["cosine_warmup"]


if __name__ == "__main__":
    pytest.main([__file__])