#!/usr/bin/env python3
"""
Test script to verify that normal ResNet models have been updated to use LayerNorm
instead of BatchNorm and have no bias in conv/linear layers.

This test validates the requirements:
1. Replace BatchNorm with LayerNorm (with no bias)  
2. Remove bias from all linear and conv layers
"""

import sys
sys.path.append('/home/runner/work/chilladam/chilladam')

import torch
import torch.nn as nn
from chilladam.models.resnet import resnet18, resnet50


def test_layernorm_and_bias_requirements():
    """Test that ResNet models meet the LayerNorm and no-bias requirements."""
    
    print("=== Testing Normal ResNet LayerNorm and No-Bias Requirements ===\n")
    
    models_to_test = [
        (resnet18, "ResNet-18"),
        (resnet50, "ResNet-50")
    ]
    
    all_tests_passed = True
    
    for model_fn, model_name in models_to_test:
        print(f"Testing {model_name}...")
        
        # Create model
        model = model_fn(num_classes=10, input_size=32)
        
        # Count layers and check requirements
        conv_with_bias = []
        linear_with_bias = []
        batchnorm_layers = []
        layernorm_count = 0
        layernorm_with_bias = []
        
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
                if module.bias is not None:
                    layernorm_with_bias.append(name)
        
        # Check requirements
        test_passed = True
        
        # Requirement 1: No BatchNorm layers
        if batchnorm_layers:
            print(f"  ❌ Found BatchNorm2d layers: {batchnorm_layers}")
            test_passed = False
        else:
            print(f"  ✅ No BatchNorm2d layers found")
        
        # Requirement 2: LayerNorm layers present
        if layernorm_count == 0:
            print(f"  ❌ No LayerNorm layers found")
            test_passed = False
        else:
            print(f"  ✅ Found {layernorm_count} LayerNorm layers")
        
        # Requirement 3: LayerNorm has no bias
        if layernorm_with_bias:
            print(f"  ❌ LayerNorm layers with bias: {layernorm_with_bias}")
            test_passed = False
        else:
            print(f"  ✅ All LayerNorm layers have bias=False")
        
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


def test_layernorm_functionality():
    """Test that LayerNorm is applied correctly to 2D feature maps."""
    
    print("=== Testing LayerNorm 2D Functionality ===\n")
    
    from chilladam.models.resnet import apply_layer_norm_2d
    
    # Create a test LayerNorm and input
    channels = 64
    layer_norm = nn.LayerNorm(channels, bias=False)
    x = torch.randn(2, channels, 8, 8)  # Batch=2, Channels=64, H=8, W=8
    
    print(f"Input shape: {x.shape}")
    
    try:
        output = apply_layer_norm_2d(x, layer_norm)
        print(f"Output shape: {output.shape}")
        
        if output.shape == x.shape:
            print("✅ LayerNorm 2D helper function works correctly")
            return True
        else:
            print(f"❌ Shape mismatch: expected {x.shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ LayerNorm 2D helper function failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Normal ResNet Models with LayerNorm Requirements\n")
    
    # Test LayerNorm functionality
    layernorm_test = test_layernorm_functionality()
    
    # Test model requirements
    requirements_test = test_layernorm_and_bias_requirements()
    
    # Final result
    print("=== FINAL TEST SUMMARY ===")
    print(f"LayerNorm functionality test: {'PASSED' if layernorm_test else 'FAILED'}")
    print(f"Model requirements test: {'PASSED' if requirements_test else 'FAILED'}")
    
    if layernorm_test and requirements_test:
        print("\n🎉 ALL TESTS PASSED! Normal ResNet models successfully updated.")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED!")
        sys.exit(1)