#!/usr/bin/env python3
"""
Test script to verify that YAT ResNet models have been updated to use RMSNorm
instead of LayerNorm.

This test validates that all YAT models now use RMSNorm instead of LayerNorm.
"""

import sys
sys.path.append('/home/runner/work/chilladam/chilladam')

import torch
import torch.nn as nn

# Import the YAT models
from chilladam.models.se_models import (
    yat_resnet18_no_se, yat_resnet34_no_se, yat_resnet50_no_se,
    BasicYATBlockNoSE, BottleneckYATBlockNoSE, apply_rms_norm_2d
)


def test_yat_rmsnorm_and_bias_requirements():
    """Test that YAT ResNet models meet the RMSNorm and no-bias requirements."""
    
    print("=== Testing YAT ResNet RMSNorm and No-Bias Requirements ===\n")
    
    models_to_test = [
        (yat_resnet18_no_se, "YAT ResNet-18 (no SE)"),
        (yat_resnet34_no_se, "YAT ResNet-34 (no SE)"),
        (yat_resnet50_no_se, "YAT ResNet-50 (no SE)")
    ]
    
    all_tests_passed = True
    
    for model_fn, model_name in models_to_test:
        print(f"Testing {model_name}...")
        
        # Create model
        model = model_fn(num_classes=10)
        
        # Count layers and check requirements
        conv_with_bias = []
        linear_with_bias = []
        batchnorm_layers = []
        layernorm_count = 0
        rmsnorm_count = 0
        dropout_count = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.bias is not None:
                    conv_with_bias.append(name)
            elif isinstance(module, nn.Linear):
                if module.bias is not None:
                    linear_with_bias.append(name)
            elif isinstance(module, nn.BatchNorm2d):
                batchnorm_layers.append(name)
            elif isinstance(module, nn.LayerNorm):
                layernorm_count += 1
            elif isinstance(module, nn.RMSNorm):
                rmsnorm_count += 1
            elif isinstance(module, nn.Dropout):
                dropout_count += 1
        
        # Check requirements
        test_passed = True
        
        # Requirement 1: No BatchNorm layers
        if batchnorm_layers:
            print(f"  ❌ Found BatchNorm2d layers: {batchnorm_layers}")
            test_passed = False
        else:
            print(f"  ✅ No BatchNorm2d layers found")
        
        # Requirement 2: No LayerNorm layers (replaced with RMSNorm)
        if layernorm_count > 0:
            print(f"  ❌ Found {layernorm_count} LayerNorm layers (should be replaced with RMSNorm)")
            test_passed = False
        else:
            print(f"  ✅ No LayerNorm layers found (replaced with RMSNorm)")
        
        # Requirement 3: RMSNorm layers present
        if rmsnorm_count == 0:
            print(f"  ❌ No RMSNorm layers found")
            test_passed = False
        else:
            print(f"  ✅ Found {rmsnorm_count} RMSNorm layers")
        
        # Requirement 4: No bias in Conv2d layers
        if conv_with_bias:
            print(f"  ❌ Conv2d layers with bias: {conv_with_bias}")
            test_passed = False
        else:
            print(f"  ✅ All Conv2d layers have bias=False")
        
        # Requirement 5: No bias in Linear layers
        if linear_with_bias:
            print(f"  ❌ Linear layers with bias: {linear_with_bias}")
            test_passed = False
        else:
            print(f"  ✅ All Linear layers have bias=False")
        
        # Test functionality
        try:
            model.eval()
            x = torch.randn(2, 3, 32, 32)
            with torch.no_grad():
                output = model(x)
            
            if output.shape == (2, 10):
                print(f"  ✅ Forward pass successful: {output.shape}")
            else:
                print(f"  ❌ Unexpected output shape: {output.shape}")
                test_passed = False
                
        except Exception as e:
            print(f"  ❌ Forward pass failed: {e}")
            test_passed = False
        
        # Final result for this model
        if test_passed:
            print(f"  🎉 {model_name} passed all requirements!\n")
        else:
            print(f"  ⚠️  {model_name} failed some requirements.\n")
            all_tests_passed = False
    
    return all_tests_passed


def test_yat_rmsnorm_functionality():
    """Test that RMSNorm is applied correctly to 2D feature maps in YAT blocks."""
    
    print("=== Testing YAT RMSNorm 2D Functionality ===\n")
    
    try:
        channels = 64
        rms_norm = nn.RMSNorm(channels, elementwise_affine=True)
        x = torch.randn(2, channels, 8, 8)  # Batch=2, Channels=64, H=8, W=8
        
        print(f"Input shape: {x.shape}")
        
        output = apply_rms_norm_2d(x, rms_norm)
        print(f"Output shape: {output.shape}")
        
        if output.shape == x.shape:
            print("✅ YAT RMSNorm 2D helper function works correctly")
            return True
        else:
            print(f"❌ Shape mismatch: expected {x.shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ YAT RMSNorm 2D helper function failed: {e}")
        return False


def test_individual_yat_blocks():
    """Test individual YAT blocks to ensure they use RMSNorm."""
    
    print("=== Testing Individual YAT Blocks ===\n")
    
    try:
        # Test BasicYATBlockNoSE
        print("Testing BasicYATBlockNoSE...")
        basic_block = BasicYATBlockNoSE(64, 64, stride=1)
        x = torch.randn(2, 64, 8, 8)
        
        basic_block.eval()
        with torch.no_grad():
            output = basic_block(x)
        
        if output.shape == x.shape:
            print("✅ BasicYATBlockNoSE forward pass successful")
        else:
            print(f"❌ BasicYATBlockNoSE shape mismatch: {output.shape} vs {x.shape}")
            return False
            
        # Check for RMSNorm
        rms_count = sum(1 for m in basic_block.modules() if isinstance(m, nn.RMSNorm))
        layer_count = sum(1 for m in basic_block.modules() if isinstance(m, nn.LayerNorm))
        
        if rms_count > 0 and layer_count == 0:
            print(f"✅ BasicYATBlockNoSE uses RMSNorm ({rms_count}) and no LayerNorm")
        else:
            print(f"❌ BasicYATBlockNoSE RMSNorm: {rms_count}, LayerNorm: {layer_count}")
            return False
        
        # Test BottleneckYATBlockNoSE
        print("\nTesting BottleneckYATBlockNoSE...")
        bottleneck_block = BottleneckYATBlockNoSE(64, 64, stride=1)
        x = torch.randn(2, 64, 8, 8)
        
        bottleneck_block.eval()
        with torch.no_grad():
            output = bottleneck_block(x)
        
        expected_shape = (2, 64 * 4, 8, 8)  # expansion = 4
        if output.shape == expected_shape:
            print("✅ BottleneckYATBlockNoSE forward pass successful")
        else:
            print(f"❌ BottleneckYATBlockNoSE shape mismatch: {output.shape} vs {expected_shape}")
            return False
            
        # Check for RMSNorm
        rms_count = sum(1 for m in bottleneck_block.modules() if isinstance(m, nn.RMSNorm))
        layer_count = sum(1 for m in bottleneck_block.modules() if isinstance(m, nn.LayerNorm))
        
        if rms_count > 0 and layer_count == 0:
            print(f"✅ BottleneckYATBlockNoSE uses RMSNorm ({rms_count}) and no LayerNorm")
        else:
            print(f"❌ BottleneckYATBlockNoSE RMSNorm: {rms_count}, LayerNorm: {layer_count}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Individual YAT block test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing YAT ResNet Models with RMSNorm Requirements\n")
    
    # Test RMSNorm functionality
    rmsnorm_test = test_yat_rmsnorm_functionality()
    
    # Test individual blocks
    blocks_test = test_individual_yat_blocks()
    
    # Test model requirements
    requirements_test = test_yat_rmsnorm_and_bias_requirements()
    
    # Final result
    print("=== FINAL TEST SUMMARY ===")
    print(f"YAT RMSNorm functionality test: {'PASSED' if rmsnorm_test else 'FAILED'}")
    print(f"YAT individual blocks test: {'PASSED' if blocks_test else 'FAILED'}")
    print(f"YAT model requirements test: {'PASSED' if requirements_test else 'FAILED'}")
    
    if rmsnorm_test and blocks_test and requirements_test:
        print("\n🎉 ALL TESTS PASSED! YAT ResNet models successfully updated with RMSNorm.")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED!")
        sys.exit(1)