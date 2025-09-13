#!/usr/bin/env python3
"""
Test script for IterableDataset support in chilladam trainer

This script tests that the trainer works correctly with IterableDataset
(streaming datasets) and doesn't fail with the len() error.
"""

import sys
import os
import torch
from torch.utils.data import IterableDataset, DataLoader
import unittest.mock as mock

# Add chilladam to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class MockIterableDataset(IterableDataset):
    """Mock IterableDataset for testing"""
    
    def __init__(self, num_samples=10, num_classes=10):
        self.num_samples = num_samples
        self.num_classes = num_classes
    
    def __iter__(self):
        for i in range(self.num_samples):
            # Create mock image data and labels
            yield {
                'pixel_values': torch.randn(3, 32, 32),  # Random RGB 32x32 image
                'label': i % self.num_classes
            }

def test_iterable_dataset_validation():
    """Test that trainer works with IterableDataset in validation"""
    
    # Mock wandb to avoid actual wandb dependency
    with mock.patch.dict('sys.modules', {'wandb': mock.MagicMock()}):
        # Import after mocking
        from chilladam import ChillAdam, resnet18
        from chilladam.training import Trainer
        
        print("Testing IterableDataset support in trainer...")
        
        # Create a model with enough classes for top-5 accuracy
        model = resnet18(num_classes=10, input_size=32)
        optimizer = ChillAdam(model.parameters())
        
        # Create trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            use_wandb=False
        )
        
        # Create mock IterableDataset and DataLoader
        mock_dataset = MockIterableDataset(num_samples=5, num_classes=10)
        val_dataloader = DataLoader(mock_dataset, batch_size=2)
        
        print("✓ Created IterableDataset and DataLoader")
        
        # Test validation epoch - this should not raise the len() error
        try:
            avg_loss, top1_acc, top5_acc = trainer.validate_epoch(val_dataloader, epoch=1)
            print(f"✓ Validation completed successfully!")
            print(f"  - Average loss: {avg_loss:.4f}")
            print(f"  - Top-1 accuracy: {top1_acc:.2f}%")
            print(f"  - Top-5 accuracy: {top5_acc:.2f}%")
            
            # Verify results are reasonable
            assert isinstance(avg_loss, float), "avg_loss should be float"
            assert isinstance(top1_acc, float), "top1_acc should be float"
            assert isinstance(top5_acc, float), "top5_acc should be float"
            assert 0 <= top1_acc <= 100, "top1_acc should be percentage"
            assert 0 <= top5_acc <= 100, "top5_acc should be percentage"
            
            print("✓ Validation results are properly formatted")
            
        except TypeError as e:
            if "object of type 'IterableDataset' has no len()" in str(e):
                print(f"❌ IterableDataset len() error still occurs: {e}")
                return False
            else:
                raise e
        
        print("✓ IterableDataset validation test passed!")
        return True

def test_regular_dataset_still_works():
    """Test that trainer still works with regular datasets"""
    
    # Mock wandb to avoid actual wandb dependency
    with mock.patch.dict('sys.modules', {'wandb': mock.MagicMock()}):
        # Import after mocking
        from chilladam import ChillAdam, resnet18
        from chilladam.training import Trainer
        from torch.utils.data import TensorDataset
        
        print("Testing regular Dataset support in trainer...")
        
        # Create a model with enough classes for top-5 accuracy
        model = resnet18(num_classes=10, input_size=32)
        optimizer = ChillAdam(model.parameters())
        
        # Create trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            use_wandb=False
        )
        
        # Create regular TensorDataset
        images = torch.randn(5, 3, 32, 32)
        labels = torch.randint(0, 10, (5,))
        dataset = TensorDataset(images, labels)
        
        # Create DataLoader that mimics the expected batch format
        def collate_fn(batch):
            images, labels = zip(*batch)
            return {
                'pixel_values': torch.stack(images),
                'label': list(labels)
            }
        
        val_dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        print("✓ Created regular Dataset and DataLoader")
        
        # Test validation epoch
        try:
            avg_loss, top1_acc, top5_acc = trainer.validate_epoch(val_dataloader, epoch=1)
            print(f"✓ Validation completed successfully!")
            print(f"  - Average loss: {avg_loss:.4f}")
            print(f"  - Top-1 accuracy: {top1_acc:.2f}%")
            print(f"  - Top-5 accuracy: {top5_acc:.2f}%")
            
        except Exception as e:
            print(f"❌ Regular dataset validation failed: {e}")
            return False
        
        print("✓ Regular Dataset validation test passed!")
        return True

if __name__ == "__main__":
    try:
        success1 = test_iterable_dataset_validation()
        success2 = test_regular_dataset_still_works()
        
        if success1 and success2:
            print("\n✅ All IterableDataset tests passed!")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)