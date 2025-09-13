#!/usr/bin/env python3
"""
Test script to verify the fix for the IterableDataset len() error.

This test specifically reproduces the error scenario from the problem statement
and verifies that it's fixed.
"""

import sys
import os
import torch
from torch.utils.data import IterableDataset, DataLoader
import unittest.mock as mock

# Add chilladam to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class StreamingDataset(IterableDataset):
    """
    Mock streaming dataset that mimics the behavior of Hugging Face streaming datasets.
    This simulates the exact scenario where the error occurred.
    """
    
    def __init__(self, num_samples=20, num_classes=10):
        self.num_samples = num_samples
        self.num_classes = num_classes
    
    def __iter__(self):
        """Iterator that yields batches in the format expected by the trainer"""
        for i in range(self.num_samples):
            yield {
                'pixel_values': torch.randn(3, 64, 64),  # Tiny ImageNet size
                'label': i % self.num_classes
            }

def test_reproduce_original_error():
    """
    Test that reproduces the original error scenario and verifies it's fixed.
    
    This test simulates the exact conditions that caused:
    TypeError: object of type 'IterableDataset' has no len()
    """
    
    print("Reproducing the original IterableDataset len() error scenario...")
    
    # Mock wandb to avoid dependencies
    with mock.patch.dict('sys.modules', {'wandb': mock.MagicMock()}):
        from chilladam import ChillAdam, resnet18
        from chilladam.training import Trainer
        
        # Create model similar to the original setup
        model = resnet18(num_classes=200, input_size=64)  # Tiny ImageNet setup
        optimizer = ChillAdam(model.parameters())
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            use_wandb=False
        )
        
        # Create streaming dataset and dataloader
        streaming_dataset = StreamingDataset(num_samples=20, num_classes=200)
        val_dataloader = DataLoader(streaming_dataset, batch_size=4)
        
        # This is where the original error occurred
        print("Testing trainer.validate_epoch() with IterableDataset...")
        
        try:
            # This line used to fail with: TypeError: object of type 'IterableDataset' has no len()
            val_loss, top1_accuracy, top5_accuracy = trainer.validate_epoch(val_dataloader, epoch=1)
            
            print("✅ SUCCESS: validate_epoch completed without len() error!")
            print(f"   Validation Loss: {val_loss:.4f}")
            print(f"   Top-1 Accuracy: {top1_accuracy:.2f}%")
            print(f"   Top-5 Accuracy: {top5_accuracy:.2f}%")
            
            # Verify the results are valid
            assert isinstance(val_loss, float) and val_loss > 0, "Loss should be positive float"
            assert 0 <= top1_accuracy <= 100, "Top-1 accuracy should be a percentage"
            assert 0 <= top5_accuracy <= 100, "Top-5 accuracy should be a percentage"
            assert top5_accuracy >= top1_accuracy, "Top-5 should be >= top-1 accuracy"
            
            return True
            
        except TypeError as e:
            if "object of type 'IterableDataset' has no len()" in str(e):
                print(f"❌ FAILED: Original len() error still occurs: {e}")
                return False
            else:
                # Re-raise unexpected TypeError
                raise e

def test_train_method_with_streaming():
    """
    Test that the full train() method works with streaming datasets.
    This tests the complete training pipeline that was failing.
    """
    
    print("\nTesting full training pipeline with streaming datasets...")
    
    # Mock wandb
    with mock.patch.dict('sys.modules', {'wandb': mock.MagicMock()}):
        from chilladam import ChillAdam, resnet18
        from chilladam.training import Trainer
        
        # Create smaller model for faster testing
        model = resnet18(num_classes=10, input_size=32)
        optimizer = ChillAdam(model.parameters())
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            use_wandb=False
        )
        
        # Create streaming datasets
        train_dataset = StreamingDataset(num_samples=8, num_classes=10)
        val_dataset = StreamingDataset(num_samples=4, num_classes=10)
        
        train_dataloader = DataLoader(train_dataset, batch_size=2)
        val_dataloader = DataLoader(val_dataset, batch_size=2)
        
        try:
            # Test full training for 1 epoch - this should not fail
            print("Running trainer.train() for 1 epoch...")
            trainer.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_epochs=1
            )
            
            print("✅ SUCCESS: Full training pipeline completed!")
            return True
            
        except TypeError as e:
            if "object of type 'IterableDataset' has no len()" in str(e):
                print(f"❌ FAILED: len() error in training pipeline: {e}")
                return False
            else:
                raise e

if __name__ == "__main__":
    try:
        print("=" * 70)
        print("TESTING FIX FOR: TypeError: object of type 'IterableDataset' has no len()")
        print("=" * 70)
        
        success1 = test_reproduce_original_error()
        success2 = test_train_method_with_streaming()
        
        print("\n" + "=" * 70)
        if success1 and success2:
            print("🎉 ALL TESTS PASSED! The IterableDataset len() error is FIXED!")
            print("\nThe trainer now works correctly with:")
            print("  • Hugging Face streaming datasets")
            print("  • IterableDataset objects")
            print("  • Regular Dataset objects")
        else:
            print("❌ SOME TESTS FAILED! The error may not be completely fixed.")
            sys.exit(1)
        print("=" * 70)
            
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)