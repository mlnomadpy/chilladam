#!/usr/bin/env python3
"""
Test script for wandb.watch() functionality in chilladam

This script tests the wandb.watch() implementation without requiring
actual wandb credentials or network access.
"""

import sys
import os
import unittest.mock as mock

# Add chilladam to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_wandb_watch_functionality():
    """Test wandb.watch() integration"""
    
    # Mock wandb to avoid actual wandb dependency for testing
    with mock.patch.dict('sys.modules', {'wandb': mock.MagicMock()}):
        # Import after mocking
        from chilladam import ChillAdam, resnet18
        from chilladam.training import Trainer
        from chilladam.config import Config, parse_args
        
        # Get mocked wandb
        mock_wandb = sys.modules['wandb']
        
        print("Testing wandb.watch() functionality...")
        
        # Test 1: Configuration defaults
        config = Config()
        assert config.wandb_watch == False
        assert config.wandb_watch_log_freq == 100
        print("✓ Configuration defaults correct")
        
        # Test 2: Trainer with wandb watch enabled
        model = resnet18(num_classes=10, input_size=32)
        optimizer = ChillAdam(model.parameters())
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            use_wandb=True,
            wandb_config={'project': 'test'},
            wandb_watch=True,
            wandb_watch_log_freq=100
        )
        
        # Verify wandb.watch was called
        mock_wandb.watch.assert_called_once_with(model, log_freq=100)
        print("✓ wandb.watch() called with correct parameters")
        
        # Test 3: Trainer with wandb watch disabled
        mock_wandb.reset_mock()
        trainer_no_watch = Trainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            use_wandb=True,
            wandb_config={'project': 'test'},
            wandb_watch=False,
            wandb_watch_log_freq=50
        )
        
        # Verify wandb.watch was NOT called
        mock_wandb.watch.assert_not_called()
        print("✓ wandb.watch() not called when disabled")
        
        print("\n✓ All wandb.watch() tests passed!")
        return True

if __name__ == "__main__":
    try:
        test_wandb_watch_functionality()
        print("\n✅ Test suite completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)