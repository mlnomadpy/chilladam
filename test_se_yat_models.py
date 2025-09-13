#!/usr/bin/env python3
"""
Test script for SE and YAT models in chilladam

This script tests the newly added Squeeze-and-Excitation and YAT model functionality.
"""

import sys
import os
import torch
import torch.nn as nn

# Add chilladam to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_se_and_yat_models():
    """Test SE and YAT model functionality"""
    
    print("Testing SE and YAT models...")
    
    # Import models
    from chilladam import (
        standard_se_resnet18, standard_se_resnet34,
        yat_resnet18, yat_resnet34
    )
    from chilladam.models import (
        SELayer, YatSELayer, BasicStandardBlock, BasicYATBlock,
        StandardConvNet, YATConvNet
    )
    
    # Test 1: SE Layer functionality
    print("\n1. Testing SE Layer...")
    se_layer = SELayer(64, reduction=16)
    x = torch.randn(2, 64, 32, 32)
    output = se_layer(x)
    assert output.shape == x.shape, f"SE layer output shape mismatch: {output.shape} vs {x.shape}"
    print("✓ SE Layer working correctly")
    
    # Test 2: YAT SE Layer functionality
    print("\n2. Testing YAT SE Layer...")
    yat_se_layer = YatSELayer(64, reduction=16)
    output = yat_se_layer(x)
    assert output.shape == x.shape, f"YAT SE layer output shape mismatch: {output.shape} vs {x.shape}"
    print("✓ YAT SE Layer working correctly")
    
    # Test 3: Basic Standard Block
    print("\n3. Testing Basic Standard Block...")
    std_block = BasicStandardBlock(64, 64, stride=1)
    output = std_block(x)
    assert output.shape == x.shape, f"Standard block output shape mismatch: {output.shape} vs {x.shape}"
    print("✓ Basic Standard Block working correctly")
    
    # Test 4: Basic YAT Block
    print("\n4. Testing Basic YAT Block...")
    yat_block = BasicYATBlock(64, 64, stride=1)
    yat_block.eval()  # Set to eval mode for consistent testing
    with torch.no_grad():
        output = yat_block(x)
    assert output.shape == x.shape, f"YAT block output shape mismatch: {output.shape} vs {x.shape}"
    print("✓ Basic YAT Block working correctly")
    
    # Test 5: Standard ConvNet models
    print("\n5. Testing Standard ConvNet models...")
    models_to_test = [
        (standard_se_resnet18, {"num_classes": 10}),
        (standard_se_resnet34, {"num_classes": 100}),
    ]
    
    for model_fn, kwargs in models_to_test:
        model = model_fn(**kwargs)
        model.eval()
        x_test = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = model(x_test)
        expected_shape = (2, kwargs["num_classes"])
        assert output.shape == expected_shape, f"Model output shape mismatch: {output.shape} vs {expected_shape}"
        print(f"✓ {model_fn.__name__} working correctly")
    
    # Test 6: YAT ConvNet models  
    print("\n6. Testing YAT ConvNet models...")
    yat_models_to_test = [
        (yat_resnet18, {"num_classes": 200}),
        (yat_resnet34, {"num_classes": 50, "use_alpha": True, "use_dropconnect": False}),
        (yat_resnet18, {"num_classes": 10, "use_alpha": False, "use_dropconnect": True, "drop_rate": 0.2}),
    ]
    
    for model_fn, kwargs in yat_models_to_test:
        model = model_fn(**kwargs)
        model.eval()
        x_test = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = model(x_test)
        expected_shape = (2, kwargs["num_classes"])
        assert output.shape == expected_shape, f"YAT model output shape mismatch: {output.shape} vs {expected_shape}"
        print(f"✓ {model_fn.__name__} with {kwargs} working correctly")
    
    # Test 7: Model creation through main.py
    print("\n7. Testing model creation through main.py...")
    sys.path.insert(0, './chilladam')
    from main import create_model
    
    main_models_to_test = [
        "standard_se_resnet18",
        "standard_se_resnet34", 
        "yat_resnet18",
        "yat_resnet34"
    ]
    
    for model_name in main_models_to_test:
        model = create_model(model_name, num_classes=10, input_size=32)
        model.eval()
        x_test = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x_test)
        assert output.shape == (1, 10), f"Main model {model_name} output shape mismatch: {output.shape} vs (1, 10)"
        print(f"✓ {model_name} creation through main.py working correctly")
    
    # Test 8: Check parameter counts and model structure
    print("\n8. Testing model structure...")
    se_model = standard_se_resnet18(num_classes=10)
    yat_model = yat_resnet18(num_classes=10)
    
    se_params = sum(p.numel() for p in se_model.parameters())
    yat_params = sum(p.numel() for p in yat_model.parameters())
    
    print(f"✓ Standard SE-ResNet-18 parameters: {se_params:,}")
    print(f"✓ YAT ResNet-18 parameters: {yat_params:,}")
    
    # Test 9: Ensure models have SE layers
    print("\n9. Testing SE layer presence...")
    
    # Check if SE layers are present in standard model
    se_layers_found = False
    for name, module in se_model.named_modules():
        if isinstance(module, SELayer):
            se_layers_found = True
            break
    assert se_layers_found, "SE layers not found in standard SE model"
    print("✓ SE layers found in standard model")
    
    # Check if YAT SE layers are present in YAT model
    yat_se_layers_found = False
    for name, module in yat_model.named_modules():
        if isinstance(module, YatSELayer):
            yat_se_layers_found = True
            break
    assert yat_se_layers_found, "YAT SE layers not found in YAT model"
    print("✓ YAT SE layers found in YAT model")
    
    print("\n✓ All SE and YAT model tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_se_and_yat_models()
        print("\n✅ SE and YAT model test suite completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)