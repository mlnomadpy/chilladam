# Test Directory

This directory contains test files and demo scripts that were moved from the root directory for better organization.

## Contents

### Demo Scripts
- `demo_scheduler.py` - Demonstrates the learning rate scheduler functionality
- `demo_lasso_regularization.py` - Shows Lasso (L1) regularization in action using loss-based approach
- `demo_loss_separation.py` - Demonstrates the new loss separation logging

### Integration Tests
- `test_normal_resnet_layernorm.py` - Test script to verify ResNet models use RMSNorm instead of LayerNorm

## Running the Scripts

All scripts can be run from the root directory of the repository:

```bash
# From repository root
python test/demo_scheduler.py
python test/demo_lasso_regularization.py  
python test/demo_loss_separation.py
python test/test_normal_resnet_layernorm.py
```

The import paths have been updated to work from this new location.