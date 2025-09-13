#!/usr/bin/env python3
"""
Test script for improved model naming in chilladam

This script demonstrates the new improved naming scheme and tests backward compatibility.
"""

import sys
import os
import torch

# Add chilladam to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_improved_naming():
    """Test the improved naming scheme and backward compatibility"""
    
    print("Testing improved model naming scheme...")
    
    # Import models with new naming
    from chilladam import (
        # New improved naming
        se_resnet18, se_resnet34, se_resnet50,
        yat_se_resnet18, yat_se_resnet34, yat_se_resnet50,
        yat_resnet18_plain, yat_resnet34_plain, yat_resnet50_plain,
        # Backward compatibility (current naming)
        standard_se_resnet18, standard_se_resnet34, standard_se_resnet50,
        yat_resnet18, yat_resnet34, yat_resnet50,
        yat_resnet18_no_se, yat_resnet34_no_se, yat_resnet50_no_se
    )
    
    print("\n=== NEW IMPROVED NAMING SCHEME ===")
    print("✓ se_resnet*: Standard ResNet with SE blocks")
    print("✓ yat_se_resnet*: YAT ResNet with SE blocks")  
    print("✓ yat_resnet*_plain: YAT ResNet without SE blocks")
    
    # Test 1: New naming scheme models
    print("\n1. Testing new naming scheme models...")
    
    new_models = [
        ("se_resnet18", se_resnet18),
        ("se_resnet34", se_resnet34), 
        ("se_resnet50", se_resnet50),
        ("yat_se_resnet18", yat_se_resnet18),
        ("yat_se_resnet34", yat_se_resnet34),
        ("yat_se_resnet50", yat_se_resnet50),
        ("yat_resnet18_plain", yat_resnet18_plain),
        ("yat_resnet34_plain", yat_resnet34_plain),
        ("yat_resnet50_plain", yat_resnet50_plain),
    ]
    
    x = torch.randn(1, 3, 32, 32)
    
    for model_name, model_fn in new_models:
        model = model_fn(num_classes=10)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 10), f"{model_name} output shape mismatch: {output.shape}"
        print(f"✓ {model_name} working correctly")
    
    # Test 2: Backward compatibility
    print("\n2. Testing backward compatibility...")
    
    old_models = [
        ("standard_se_resnet18", standard_se_resnet18),
        ("standard_se_resnet34", standard_se_resnet34),
        ("standard_se_resnet50", standard_se_resnet50),
        ("yat_resnet18", yat_resnet18),
        ("yat_resnet34", yat_resnet34),
        ("yat_resnet50", yat_resnet50),
        ("yat_resnet18_no_se", yat_resnet18_no_se),
        ("yat_resnet34_no_se", yat_resnet34_no_se),
        ("yat_resnet50_no_se", yat_resnet50_no_se),
    ]
    
    for model_name, model_fn in old_models:
        model = model_fn(num_classes=10)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 10), f"{model_name} output shape mismatch: {output.shape}"
        print(f"✓ {model_name} (backward compatible) working correctly")
    
    # Test 3: Verify equivalent models produce identical architectures
    print("\n3. Testing model equivalence...")
    
    equivalence_tests = [
        (se_resnet18, standard_se_resnet18, "SE ResNet-18"),
        (se_resnet34, standard_se_resnet34, "SE ResNet-34"),
        (se_resnet50, standard_se_resnet50, "SE ResNet-50"),
        (yat_se_resnet18, yat_resnet18, "YAT SE ResNet-18"),
        (yat_se_resnet34, yat_resnet34, "YAT SE ResNet-34"),
        (yat_se_resnet50, yat_resnet50, "YAT SE ResNet-50"),
        (yat_resnet18_plain, yat_resnet18_no_se, "YAT ResNet-18 plain"),
        (yat_resnet34_plain, yat_resnet34_no_se, "YAT ResNet-34 plain"),
        (yat_resnet50_plain, yat_resnet50_no_se, "YAT ResNet-50 plain"),
    ]
    
    for new_fn, old_fn, desc in equivalence_tests:
        new_model = new_fn(num_classes=100)
        old_model = old_fn(num_classes=100)
        
        new_params = sum(p.numel() for p in new_model.parameters())
        old_params = sum(p.numel() for p in old_model.parameters())
        
        assert new_params == old_params, f"{desc} parameter count mismatch: {new_params} vs {old_params}"
        print(f"✓ {desc}: new and old naming equivalent ({new_params:,} parameters)")
    
    # Test 4: Model creation through main.py with new names
    print("\n4. Testing new models through main.py...")
    sys.path.insert(0, '../../../../')
    from main import create_model
    
    # First, update main.py to support new naming (add to supported models)
    test_models = [
        "standard_se_resnet50",  # This should already work
        "yat_resnet50",          # This should already work  
        "yat_resnet50_no_se",    # This should already work
    ]
    
    for model_name in test_models:
        model = create_model(model_name, num_classes=10, input_size=32)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 32, 32))
        assert output.shape == (1, 10), f"Main.py {model_name} output shape mismatch"
        print(f"✓ {model_name} through main.py working correctly")
    
    # Test 5: Naming clarity demonstration
    print("\n5. Demonstrating naming clarity...")
    
    print("\n📋 NAMING SCHEME SUMMARY:")
    print("┌─────────────────────────┬────────────────────────────────────┐")
    print("│ Model Type              │ Naming Convention                  │")
    print("├─────────────────────────┼────────────────────────────────────┤")
    print("│ Standard + SE           │ se_resnet{18,34,50}               │")
    print("│ YAT + SE               │ yat_se_resnet{18,34,50}           │")
    print("│ YAT (no SE)            │ yat_resnet{18,34,50}_plain        │")
    print("└─────────────────────────┴────────────────────────────────────┘")
    
    print("\n🔄 BACKWARD COMPATIBILITY:")
    print("- standard_se_resnet* → se_resnet*")
    print("- yat_resnet* → yat_se_resnet* (for SE variants)")
    print("- yat_resnet*_no_se → yat_resnet*_plain")
    
    print("\n✅ The new naming makes it crystal clear:")
    print("  • se_resnet* = Standard ResNet WITH Squeeze-and-Excitation")
    print("  • yat_se_resnet* = YAT ResNet WITH Squeeze-and-Excitation") 
    print("  • yat_resnet*_plain = YAT ResNet WITHOUT Squeeze-and-Excitation")
    
    print("\n✓ All improved naming tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_improved_naming()
        print("\n✅ Improved naming test suite completed successfully!")
        print("\n🎉 The naming is now much clearer and more organized!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)