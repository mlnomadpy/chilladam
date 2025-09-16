"""
Test enhanced scheduler functionality with restart behavior and linear decay.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add the chilladam package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from chilladam.schedulers import create_scheduler
from chilladam.schedulers.cosine_warmup import CosineAnnealingWarmupScheduler
from chilladam.optimizers import create_optimizer


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


def test_cosine_warmup_restart_behavior():
    """Test cosine warmup scheduler restart behavior."""
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        total_epochs=6,
        warmup_epochs=2,
        eta_min=1e-6,
        restart=True
    )
    
    # Record learning rates for two complete cycles
    lrs = []
    for epoch in range(12):  # Two complete cycles
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)
        scheduler.step()
    
    # Check that the pattern repeats after total_epochs
    cycle1_lrs = lrs[:6]
    cycle2_lrs = lrs[6:12]
    
    # Learning rates should be identical between cycles (allowing for small floating point differences)
    for i, (lr1, lr2) in enumerate(zip(cycle1_lrs, cycle2_lrs)):
        assert abs(lr1 - lr2) < 1e-6, f"Epoch {i}: Cycle 1 LR {lr1} != Cycle 2 LR {lr2}"


def test_cosine_warmup_no_restart():
    """Test cosine warmup scheduler without restart."""
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        total_epochs=6,
        warmup_epochs=2,
        eta_min=1e-6,
        restart=False
    )
    
    # Record learning rates beyond total_epochs
    lrs = []
    for epoch in range(10):
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)
        scheduler.step()
    
    # After total_epochs, should continue cosine annealing (not restart)
    # The LR at epoch 6 should be different from epoch 0
    assert abs(lrs[6] - lrs[0]) > 1e-3, "Should not restart when restart=False"


def test_linear_decay_phase():
    """Test cosine warmup scheduler with linear decay phase."""
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        total_epochs=8,
        warmup_epochs=2,
        eta_min=0.01,
        linear_decay_epochs=3,
        final_lr=1e-6,
        restart=True
    )
    
    # Record learning rates
    lrs = []
    for epoch in range(8):
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)
        scheduler.step()
    
    # Check phases
    # Warmup: epochs 0-1
    assert lrs[0] < lrs[1], "Learning rate should increase during warmup"
    
    # Cosine: epochs 2-4 (3 epochs)
    # Linear decay: epochs 5-7 (3 epochs)
    assert lrs[5] > lrs[6] > lrs[7], "Learning rate should decrease during linear decay"
    
    # Final LR should approach final_lr
    assert abs(lrs[7] - 1e-6) < 1e-5, f"Final LR should be close to final_lr, got {lrs[7]}"


def test_linear_decay_with_restart():
    """Test linear decay phase respects restart behavior."""
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        total_epochs=5,
        warmup_epochs=1,
        eta_min=0.01,
        linear_decay_epochs=2,
        final_lr=1e-6,
        restart=True
    )
    
    # Record learning rates for two cycles
    lrs = []
    for epoch in range(10):  # Two complete cycles
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)
        scheduler.step()
    
    # Check that decay phase repeats in second cycle
    cycle1_decay_end = lrs[4]  # End of first cycle decay
    cycle2_decay_end = lrs[9]  # End of second cycle decay
    
    assert abs(cycle1_decay_end - cycle2_decay_end) < 1e-6, \
        f"Decay end should repeat: {cycle1_decay_end} vs {cycle2_decay_end}"


def test_factory_creates_enhanced_scheduler():
    """Test that factory creates scheduler with new parameters."""
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    scheduler = create_scheduler(
        "cosine_warmup",
        optimizer,
        total_epochs=10,
        warmup_epochs=3,
        eta_min=1e-6,
        linear_decay_epochs=2,
        final_lr=1e-7,
        restart=False
    )
    
    assert scheduler is not None
    assert isinstance(scheduler, CosineAnnealingWarmupScheduler)
    assert scheduler.total_epochs == 10
    assert scheduler.warmup_epochs == 3
    assert scheduler.linear_decay_epochs == 2
    assert scheduler.final_lr == 1e-7
    assert scheduler.restart == False


def test_scheduler_parameter_validation():
    """Test that scheduler validates parameters correctly."""
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Should raise error if warmup + decay > total
    with pytest.raises(ValueError):
        CosineAnnealingWarmupScheduler(
            optimizer,
            total_epochs=5,
            warmup_epochs=3,
            linear_decay_epochs=3  # 3 + 3 > 5
        )


def test_default_final_lr_behavior():
    """Test that final_lr defaults to eta_min when not specified."""
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        total_epochs=6,
        warmup_epochs=1,
        eta_min=0.01,
        linear_decay_epochs=2,
        final_lr=None  # Should default to eta_min
    )
    
    assert scheduler.final_lr == scheduler.eta_min


def test_integration_with_different_optimizers():
    """Test enhanced scheduler works with different optimizers."""
    model = SimpleModel()
    
    optimizers_to_test = [
        ("adam", torch.optim.Adam(model.parameters(), lr=0.1)),
        ("sgd", torch.optim.SGD(model.parameters(), lr=0.1)),
    ]
    
    for opt_name, optimizer in optimizers_to_test:
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer,
            total_epochs=5,
            warmup_epochs=1,
            linear_decay_epochs=1,
            restart=True
        )
        
        # Should work without errors
        for _ in range(10):
            scheduler.step()
        
        # Learning rate should be valid
        lr = optimizer.param_groups[0]['lr']
        assert lr >= 0, f"Learning rate should be non-negative for {opt_name}, got {lr}"


def test_scheduler_with_chilladam():
    """Test enhanced scheduler integration with ChillAdam."""
    model = SimpleModel()
    
    optimizer = create_optimizer(
        "chilladam",
        model.parameters(),
        min_lr=1e-5,
        max_lr=0.1
    )
    
    scheduler = create_scheduler(
        "cosine_warmup",
        optimizer,
        total_epochs=8,
        warmup_epochs=2,
        linear_decay_epochs=2,
        final_lr=1e-6,
        restart=True
    )
    
    # Should work without errors
    assert scheduler is not None
    for _ in range(16):  # Two complete cycles
        scheduler.step()
    
    # Learning rate should be valid
    lr = optimizer.param_groups[0]['lr']
    assert lr >= 0, f"Learning rate should be non-negative, got {lr}"


def test_edge_case_zero_phases():
    """Test scheduler with zero-length phases."""
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Zero warmup epochs
    scheduler1 = CosineAnnealingWarmupScheduler(
        optimizer,
        total_epochs=5,
        warmup_epochs=0,
        linear_decay_epochs=0
    )
    
    # Should work (pure cosine)
    for _ in range(5):
        scheduler1.step()
    
    # Zero cosine epochs (only warmup + decay)
    scheduler2 = CosineAnnealingWarmupScheduler(
        optimizer,
        total_epochs=3,
        warmup_epochs=2,
        linear_decay_epochs=1
    )
    
    # Should work
    for _ in range(3):
        scheduler2.step()


if __name__ == "__main__":
    pytest.main([__file__])