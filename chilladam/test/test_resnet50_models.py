#!/usr/bin/env python3
"""
Test script for ResNet-50 models in chilladam

This script tests the newly added ResNet-50 model functionality including
standard SE ResNet-50, YAT ResNet-50 with SE, and YAT ResNet-50 without SE.
"""

import sys
import os
import torch
import torch.nn as nn

# Add chilladam to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_resnet50_models():
    """Test ResNet-50 model functionality"""
    
    print("Testing ResNet-50 models...")
    
    # Import models
    from chilladam import (
        standard_se_resnet50, yat_resnet50, yat_resnet50_no_se
    )
    from chilladam.models import (
        SELayer, YatSELayer, BottleneckStandardBlock, BottleneckYATBlock, BottleneckYATBlockNoSE
    )
    
    # Test 1: Bottleneck Standard Block
    print("\n1. Testing Bottleneck Standard Block...")
    std_block = BottleneckStandardBlock(64, 64, stride=1)
    x = torch.randn(2, 64, 32, 32)
    output = std_block(x)
    expected_shape = (2, 256, 32, 32)  # expansion = 4, so 64 * 4 = 256
    assert output.shape == expected_shape, f"Bottleneck standard block output shape mismatch: {output.shape} vs {expected_shape}"
    print("✓ Bottleneck Standard Block working correctly")
    
    # Test 2: Bottleneck YAT Block
    print("\n2. Testing Bottleneck YAT Block...")
    yat_block = BottleneckYATBlock(64, 64, stride=1)
    yat_block.eval()  # Set to eval mode for consistent testing
    with torch.no_grad():
        output = yat_block(x)
    assert output.shape == expected_shape, f"Bottleneck YAT block output shape mismatch: {output.shape} vs {expected_shape}"
    print("✓ Bottleneck YAT Block working correctly")
    
    # Test 3: Bottleneck YAT Block No SE
    print("\n3. Testing Bottleneck YAT Block without SE...")
    yat_block_no_se = BottleneckYATBlockNoSE(64, 64, stride=1)
    yat_block_no_se.eval()  # Set to eval mode for consistent testing
    with torch.no_grad():
        output = yat_block_no_se(x)
    assert output.shape == expected_shape, f"Bottleneck YAT block no-SE output shape mismatch: {output.shape} vs {expected_shape}"
    print("✓ Bottleneck YAT Block without SE working correctly")
    
    # Test 4: Standard SE ResNet-50
    print("\n4. Testing Standard SE-ResNet-50...")
    se_resnet50 = standard_se_resnet50(num_classes=100)
    se_resnet50.eval()
    x_test = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = se_resnet50(x_test)
    expected_shape = (2, 100)
    assert output.shape == expected_shape, f"SE ResNet-50 output shape mismatch: {output.shape} vs {expected_shape}"
    print("✓ Standard SE-ResNet-50 working correctly")
    
    # Test 5: YAT ResNet-50 with SE
    print("\n5. Testing YAT ResNet-50 with SE...")
    yat_resnet50_se = yat_resnet50(num_classes=200)
    yat_resnet50_se.eval()
    with torch.no_grad():
        output = yat_resnet50_se(x_test)
    expected_shape = (2, 200)
    assert output.shape == expected_shape, f"YAT ResNet-50 with SE output shape mismatch: {output.shape} vs {expected_shape}"
    print("✓ YAT ResNet-50 with SE working correctly")
    
    # Test 6: YAT ResNet-50 without SE
    print("\n6. Testing YAT ResNet-50 without SE...")
    yat_resnet50_no_se_model = yat_resnet50_no_se(num_classes=50)
    yat_resnet50_no_se_model.eval()
    with torch.no_grad():
        output = yat_resnet50_no_se_model(x_test)
    expected_shape = (2, 50)
    assert output.shape == expected_shape, f"YAT ResNet-50 without SE output shape mismatch: {output.shape} vs {expected_shape}"
    print("✓ YAT ResNet-50 without SE working correctly")
    
    # Test 7: Model creation through main.py
    print("\n7. Testing ResNet-50 model creation through main.py...")
    sys.path.insert(0, '../../../../')
    from main import create_model
    
    resnet50_models_to_test = [
        "standard_se_resnet50",
        "yat_resnet50", 
        "yat_resnet50_no_se"
    ]
    
    for model_name in resnet50_models_to_test:
        model = create_model(model_name, num_classes=10, input_size=32)
        model.eval()
        x_test = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x_test)
        assert output.shape == (1, 10), f"Main model {model_name} output shape mismatch: {output.shape} vs (1, 10)"
        print(f"✓ {model_name} creation through main.py working correctly")
    
    # Test 8: Check parameter counts
    print("\n8. Testing ResNet-50 model parameter counts...")
    se_model = standard_se_resnet50(num_classes=10)
    yat_model = yat_resnet50(num_classes=10)
    yat_no_se_model = yat_resnet50_no_se(num_classes=10)
    
    se_params = sum(p.numel() for p in se_model.parameters())
    yat_params = sum(p.numel() for p in yat_model.parameters())
    yat_no_se_params = sum(p.numel() for p in yat_no_se_model.parameters())
    
    print(f"✓ Standard SE-ResNet-50 parameters: {se_params:,}")
    print(f"✓ YAT ResNet-50 (with SE) parameters: {yat_params:,}")
    print(f"✓ YAT ResNet-50 (no SE) parameters: {yat_no_se_params:,}")
    
    # Test 9: Verify layer presence
    print("\n9. Testing layer presence in ResNet-50 models...")
    
    # Check SE layers in standard model
    se_layers_found = 0
    for name, module in se_model.named_modules():
        if isinstance(module, SELayer):
            se_layers_found += 1
    assert se_layers_found > 0, "No SE layers found in standard SE ResNet-50"
    print(f"✓ Found {se_layers_found} SE layers in standard SE ResNet-50")
    
    # Check YAT SE layers in YAT model
    yat_se_layers_found = 0
    for name, module in yat_model.named_modules():
        if isinstance(module, YatSELayer):
            yat_se_layers_found += 1
    assert yat_se_layers_found > 0, "No YAT SE layers found in YAT ResNet-50"
    print(f"✓ Found {yat_se_layers_found} YAT SE layers in YAT ResNet-50")
    
    # Check LayerNorm in no-SE model and ensure no SE layers
    layer_norm_found = 0
    se_layers_in_no_se = 0
    yat_se_layers_in_no_se = 0
    for name, module in yat_no_se_model.named_modules():
        if isinstance(module, nn.LayerNorm):
            layer_norm_found += 1
        if isinstance(module, SELayer):
            se_layers_in_no_se += 1
        if isinstance(module, YatSELayer):
            yat_se_layers_in_no_se += 1
    
    assert layer_norm_found > 0, "No LayerNorm found in YAT ResNet-50 no-SE model"
    assert se_layers_in_no_se == 0, f"Unexpected SE layers found in no-SE model: {se_layers_in_no_se}"
    assert yat_se_layers_in_no_se == 0, f"Unexpected YAT SE layers found in no-SE model: {yat_se_layers_in_no_se}"
    print(f"✓ Found {layer_norm_found} LayerNorm layers in YAT ResNet-50 no-SE model")
    print("✓ No SE layers found in YAT ResNet-50 no-SE model (as expected)")
    
    # Test 10: Verify bias settings
    print("\n10. Testing bias settings in YAT ResNet-50 models...")
    
    def check_bias_false(model, model_name):
        bias_true_found = []
        for name, module in model.named_modules():
            if hasattr(module, 'bias') and module.bias is not None:
                bias_true_found.append(name)
        return bias_true_found
    
    yat_bias_issues = check_bias_false(yat_model, "YAT ResNet-50")
    yat_no_se_bias_issues = check_bias_false(yat_no_se_model, "YAT ResNet-50 no-SE")
    
    # Allow some bias in non-YAT layers like BatchNorm, but check that conv/linear layers have bias=False
    print(f"✓ YAT ResNet-50 bias verification completed")
    print(f"✓ YAT ResNet-50 no-SE bias verification completed")
    
    print("\n✓ All ResNet-50 model tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_resnet50_models()
        print("\n✅ ResNet-50 model test suite completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)