#!/usr/bin/env python3
"""
Simple test to verify the IterableDataset len() fix works.
"""

import sys
import os
import torch
from torch.utils.data import IterableDataset, DataLoader
import unittest.mock as mock

# Add chilladam to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class StreamingDataset(IterableDataset):
    """Mock streaming dataset that reproduces the original error scenario"""
    
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
    
    def __iter__(self):
        for i in range(self.num_samples):
            yield {
                'pixel_values': torch.randn(3, 64, 64),
                'label': i % 10
            }

def test_fix():
    """Test that the validation works with IterableDataset"""
    
    print("Testing IterableDataset validation fix...")
    
    # Mock wandb
    with mock.patch.dict('sys.modules', {'wandb': mock.MagicMock()}):
        from chilladam import ChillAdam, resnet18
        from chilladam.training import Trainer
        
        # Setup
        model = resnet18(num_classes=10, input_size=64)
        optimizer = ChillAdam(model.parameters())
        trainer = Trainer(model=model, optimizer=optimizer, device='cpu', use_wandb=False)
        
        # Create IterableDataset (this was causing the len() error)
        dataset = StreamingDataset(num_samples=8)
        dataloader = DataLoader(dataset, batch_size=2)
        
        # This should work without the len() error
        val_loss, top1_acc, top5_acc = trainer.validate_epoch(dataloader, epoch=1)
        
        print(f"✅ SUCCESS! Validation completed:")
        print(f"   Loss: {val_loss:.4f}")
        print(f"   Top-1: {top1_acc:.2f}%")
        print(f"   Top-5: {top5_acc:.2f}%")
        
        return True

if __name__ == "__main__":
    try:
        if test_fix():
            print("\n🎉 Fix verified! The IterableDataset len() error is resolved.")
    except Exception as e:
        if "object of type 'IterableDataset' has no len()" in str(e):
            print(f"❌ Fix failed: {e}")
        else:
            print(f"❌ Unexpected error: {e}")
        sys.exit(1)