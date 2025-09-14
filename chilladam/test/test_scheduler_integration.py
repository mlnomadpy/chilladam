"""
Integration test for scheduler functionality with training.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add the chilladam package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from chilladam.training import Trainer
from chilladam.optimizers import create_optimizer
from chilladam.schedulers import create_scheduler


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_size=10, output_size=10):  # Changed to 10 classes for top-5 testing
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)


class SimpleDataLoader:
    """Simple data loader for testing."""
    def __init__(self, batch_size=4, num_batches=3):
        self.batch_size = batch_size
        self.num_batches = num_batches
        
    def __iter__(self):
        for _ in range(self.num_batches):
            yield {
                "pixel_values": torch.randn(self.batch_size, 10),
                "label": torch.randint(0, 10, (self.batch_size,))  # Changed to 10 classes
            }
    
    def __len__(self):
        return self.num_batches


def test_trainer_with_cosine_scheduler():
    """Test trainer with cosine scheduler integration."""
    model = SimpleModel()
    
    # Create optimizer
    optimizer = create_optimizer(
        "adam",
        model.parameters(),
        lr=0.1
    )
    
    # Create cosine scheduler
    scheduler = create_scheduler(
        "cosine",
        optimizer,
        t_max=5,
        eta_min=1e-6
    )
    
    # Create trainer with scheduler
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu"
    )
    
    # Create simple data
    train_loader = SimpleDataLoader()
    val_loader = SimpleDataLoader()
    
    # Record initial learning rate
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Train for a few epochs
    for epoch in range(3):
        train_loss = trainer.train_epoch(train_loader, epoch + 1)
        val_loss, top1_acc, top5_acc = trainer.validate_epoch(val_loader, epoch + 1)
        
        # Check that training runs without errors
        assert isinstance(train_loss, float)
        assert isinstance(val_loss, float)
        assert isinstance(top1_acc, float)
        assert isinstance(top5_acc, float)
    
    # Learning rate should have changed
    final_lr = optimizer.param_groups[0]['lr']
    assert final_lr != initial_lr


def test_trainer_with_chilladam_scheduler():
    """Test trainer with ChillAdam and cosine scheduler integration."""
    model = SimpleModel()
    
    # Create ChillAdam optimizer
    optimizer = create_optimizer(
        "chilladam",
        model.parameters(),
        min_lr=1e-5,
        max_lr=1.0
    )
    
    # Create cosine scheduler
    scheduler = create_scheduler(
        "cosine",
        optimizer,
        t_max=5,
        eta_min=1e-6
    )
    
    # Create trainer with scheduler
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu"
    )
    
    # Create simple data
    train_loader = SimpleDataLoader()
    val_loader = SimpleDataLoader()
    
    # Record initial learning rate
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Train for a few epochs
    for epoch in range(3):
        train_loss = trainer.train_epoch(train_loader, epoch + 1)
        val_loss, top1_acc, top5_acc = trainer.validate_epoch(val_loader, epoch + 1)
        
        # Check that training runs without errors
        assert isinstance(train_loss, float)
        assert isinstance(val_loss, float)
        assert isinstance(top1_acc, float)
        assert isinstance(top5_acc, float)
    
    # Learning rate should have changed
    final_lr = optimizer.param_groups[0]['lr']
    assert final_lr != initial_lr


def test_trainer_without_scheduler():
    """Test trainer without scheduler (backward compatibility)."""
    model = SimpleModel()
    
    # Create optimizer
    optimizer = create_optimizer(
        "adam",
        model.parameters(),
        lr=0.1
    )
    
    # Create trainer without scheduler
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,  # No scheduler
        device="cpu"
    )
    
    # Create simple data
    train_loader = SimpleDataLoader()
    val_loader = SimpleDataLoader()
    
    # Record initial learning rate
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Train for a few epochs
    for epoch in range(3):
        train_loss = trainer.train_epoch(train_loader, epoch + 1)
        val_loss, top1_acc, top5_acc = trainer.validate_epoch(val_loader, epoch + 1)
        
        # Check that training runs without errors
        assert isinstance(train_loss, float)
        assert isinstance(val_loss, float)
        assert isinstance(top1_acc, float)
        assert isinstance(top5_acc, float)
    
    # Learning rate should remain the same (no scheduler)
    final_lr = optimizer.param_groups[0]['lr']
    assert final_lr == initial_lr


def test_different_schedulers():
    """Test different scheduler types work with trainer."""
    model = SimpleModel()
    
    schedulers_to_test = [
        ("step", {"step_size": 2, "gamma": 0.1}),
        ("exponential", {"gamma": 0.95}),
        ("cosine", {"t_max": 5, "eta_min": 1e-6})
    ]
    
    for scheduler_name, scheduler_kwargs in schedulers_to_test:
        # Create fresh optimizer for each test
        optimizer = create_optimizer(
            "adam",
            model.parameters(),
            lr=0.1
        )
        
        # Create scheduler
        scheduler = create_scheduler(
            scheduler_name,
            optimizer,
            **scheduler_kwargs
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cpu"
        )
        
        # Test one epoch
        train_loader = SimpleDataLoader()
        train_loss = trainer.train_epoch(train_loader, 1)
        
        # Should run without errors
        assert isinstance(train_loss, float)


if __name__ == "__main__":
    pytest.main([__file__])