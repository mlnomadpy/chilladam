#!/usr/bin/env python3
"""
Test script to verify that ResNet models have been updated to use RMSNorm
instead of LayerNorm and include dropout functionality.

This test validates the requirements:
1. Replace LayerNorm with RMSNorm  
2. Add dropout functionality
3. Remove bias from all linear and conv layers
"""

import sys
sys.path.append('/home/runner/work/chilladam/chilladam')

import torch
import torch.nn as nn

# Import the resnet module directly to avoid __init__ dependencies
import importlib.util

# Load resnet.py directly
resnet_path = '/home/runner/work/chilladam/chilladam/chilladam/models/resnet.py'
spec = importlib.util.spec_from_file_location("resnet", resnet_path)
resnet_module = importlib.util.module_from_spec(spec)
sys.modules["resnet"] = resnet_module
spec.loader.exec_module(resnet_module)

resnet18 = resnet_module.resnet18
resnet50 = resnet_module.resnet50
apply_rms_norm_2d = resnet_module.apply_rms_norm_2d


def test_rmsnorm_and_bias_requirements():
    """Test that ResNet models meet the RMSNorm and no-bias requirements."""
    
    print("=== Testing ResNet RMSNorm and No-Bias Requirements ===\n")
    
    models_to_test = [
        (resnet18, "ResNet-18"),
        (resnet50, "ResNet-50")
    ]
    
    all_tests_passed = True
    
    for model_fn, model_name in models_to_test:
        print(f"Testing {model_name}...")
        
        # Create model
        model = model_fn(num_classes=10, input_size=32, dropout_rate=0.1)
        
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
        
        # Requirement 4: Dropout layers present
        if dropout_count == 0:
            print(f"  ❌ No Dropout layers found")
            test_passed = False
        else:
            print(f"  ✅ Found {dropout_count} Dropout layers")
        
        # Requirement 5: No bias in Conv2d layers
        if conv_with_bias:
            print(f"  ❌ Conv2d layers with bias: {conv_with_bias}")
            test_passed = False
        else:
            print(f"  ✅ All Conv2d layers have bias=False")
        
        # Requirement 6: No bias in Linear layers
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


def test_rmsnorm_functionality():
    """Test that RMSNorm is applied correctly to 2D feature maps."""
    
    print("=== Testing RMSNorm 2D Functionality ===\n")
    
    try:
        channels = 64
        rms_norm = nn.RMSNorm(channels, elementwise_affine=True)
        x = torch.randn(2, channels, 8, 8)  # Batch=2, Channels=64, H=8, W=8
        
        print(f"Input shape: {x.shape}")
        
        output = apply_rms_norm_2d(x, rms_norm)
        print(f"Output shape: {output.shape}")
        
        if output.shape == x.shape:
            print("✅ RMSNorm 2D helper function works correctly")
            return True
        else:
            print(f"❌ Shape mismatch: expected {x.shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ RMSNorm 2D helper function failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing ResNet Models with RMSNorm and Dropout Requirements\n")
    
    # Test RMSNorm functionality
    rmsnorm_test = test_rmsnorm_functionality()
    
    # Test model requirements
    requirements_test = test_rmsnorm_and_bias_requirements()
    
    # Final result
    print("=== FINAL TEST SUMMARY ===")
    print(f"RMSNorm functionality test: {'PASSED' if rmsnorm_test else 'FAILED'}")
    print(f"Model requirements test: {'PASSED' if requirements_test else 'FAILED'}")
    
    if rmsnorm_test and requirements_test:
        print("\n🎉 ALL TESTS PASSED! ResNet models successfully updated with RMSNorm and dropout.")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED!")
        sys.exit(1)