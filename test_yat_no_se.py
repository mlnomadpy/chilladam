#!/usr/bin/env python3
"""
Test script for YAT models without SE layers

This script tests the newly added YAT models without Squeeze-and-Excitation layers.
"""

import sys
import os
import torch
import torch.nn as nn

# Add chilladam to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_yat_no_se_models():
    """Test YAT model functionality without SE layers"""
    
    print("Testing YAT models without SE layers...")
    
    # Import models
    from chilladam import (
        yat_resnet18_no_se, yat_resnet34_no_se
    )
    from chilladam.models import (
        BasicYATBlockNoSE, YatSELayer, SELayer
    )
    
    # Test 1: Basic YAT Block without SE functionality
    print("\n1. Testing Basic YAT Block without SE...")
    yat_block_no_se = BasicYATBlockNoSE(64, 64, stride=1)
    x = torch.randn(2, 64, 32, 32)
    yat_block_no_se.eval()  # Set to eval mode for consistent testing
    with torch.no_grad():
        output = yat_block_no_se(x)
    assert output.shape == x.shape, f"YAT block no-SE output shape mismatch: {output.shape} vs {x.shape}"
    print("✓ Basic YAT Block without SE working correctly")
    
    # Test 2: Verify LayerNorm is applied
    print("\n2. Testing LayerNorm presence...")
    layer_norm_found = False
    for name, module in yat_block_no_se.named_modules():
        if isinstance(module, nn.LayerNorm):
            layer_norm_found = True
            break
    assert layer_norm_found, "LayerNorm not found in YAT block without SE"
    print("✓ LayerNorm found in YAT block without SE")
    
    # Test 3: Verify no SE layers are present
    print("\n3. Testing absence of SE layers...")
    se_layer_found = False
    yat_se_layer_found = False
    for name, module in yat_block_no_se.named_modules():
        if isinstance(module, SELayer):
            se_layer_found = True
        if isinstance(module, YatSELayer):
            yat_se_layer_found = True
    assert not se_layer_found, "Unexpected SE layer found in no-SE block"
    assert not yat_se_layer_found, "Unexpected YAT SE layer found in no-SE block"
    print("✓ No SE layers found in YAT block without SE (as expected)")
    
    # Test 4: YAT ConvNet models without SE
    print("\n4. Testing YAT ConvNet models without SE...")
    no_se_models_to_test = [
        (yat_resnet18_no_se, {"num_classes": 200}),
        (yat_resnet34_no_se, {"num_classes": 50, "use_alpha": True, "use_dropconnect": False}),
        (yat_resnet18_no_se, {"num_classes": 10, "use_alpha": False, "use_dropconnect": True, "drop_rate": 0.2}),
    ]
    
    for model_fn, kwargs in no_se_models_to_test:
        model = model_fn(**kwargs)
        model.eval()
        x_test = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = model(x_test)
        expected_shape = (2, kwargs["num_classes"])
        assert output.shape == expected_shape, f"No-SE YAT model output shape mismatch: {output.shape} vs {expected_shape}"
        print(f"✓ {model_fn.__name__} with {kwargs} working correctly")
    
    # Test 5: Model creation through main.py
    print("\n5. Testing model creation through main.py...")
    sys.path.insert(0, './chilladam')
    from main import create_model
    
    main_models_to_test = [
        "yat_resnet18_no_se",
        "yat_resnet34_no_se"
    ]
    
    for model_name in main_models_to_test:
        model = create_model(model_name, num_classes=10, input_size=32)
        model.eval()
        x_test = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x_test)
        assert output.shape == (1, 10), f"Main model {model_name} output shape mismatch: {output.shape} vs (1, 10)"
        print(f"✓ {model_name} creation through main.py working correctly")
    
    # Test 6: Compare parameter counts
    print("\n6. Testing model structure and parameter counts...")
    se_model = yat_resnet18_no_se(num_classes=10)
    
    se_params = sum(p.numel() for p in se_model.parameters())
    print(f"✓ YAT ResNet-18 (no SE) parameters: {se_params:,}")
    
    # Test 7: Ensure models have LayerNorm but no SE layers
    print("\n7. Testing LayerNorm presence and SE absence in full models...")
    
    # Check if LayerNorm layers are present in no-SE model
    layer_norm_layers_found = False
    se_layers_found = False
    yat_se_layers_found = False
    
    for name, module in se_model.named_modules():
        if isinstance(module, nn.LayerNorm):
            layer_norm_layers_found = True
        if isinstance(module, SELayer):
            se_layers_found = True
        if isinstance(module, YatSELayer):
            yat_se_layers_found = True
            
    assert layer_norm_layers_found, "LayerNorm layers not found in no-SE YAT model"
    assert not se_layers_found, "Unexpected SE layers found in no-SE YAT model"
    assert not yat_se_layers_found, "Unexpected YAT SE layers found in no-SE YAT model"
    
    print("✓ LayerNorm layers found in no-SE YAT model")
    print("✓ No SE layers found in no-SE YAT model (as expected)")
    print("✓ No YAT SE layers found in no-SE YAT model (as expected)")
    
    # Test 8: Verify different behavior with/without SE
    print("\n8. Comparing models with and without SE...")
    from chilladam import yat_resnet18
    
    se_model_18 = yat_resnet18(num_classes=10)
    no_se_model_18 = yat_resnet18_no_se(num_classes=10)
    
    se_params = sum(p.numel() for p in se_model_18.parameters())
    no_se_params = sum(p.numel() for p in no_se_model_18.parameters())
    
    print(f"✓ YAT ResNet-18 (with SE) parameters: {se_params:,}")
    print(f"✓ YAT ResNet-18 (no SE) parameters: {no_se_params:,}")
    
    # The no-SE model should have fewer parameters due to no SE layers
    # but might have more due to LayerNorm - let's just ensure they're different
    assert se_params != no_se_params, "Parameter counts should be different between SE and no-SE models"
    print("✓ Parameter counts differ between SE and no-SE models as expected")
    
    print("\n✓ All YAT no-SE model tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_yat_no_se_models()
        print("\n✅ YAT no-SE model test suite completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)