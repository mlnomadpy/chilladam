"""
Test suite for Vision Transformer (ViT) models.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add the chilladam package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from chilladam.models import vit_base, vit_large, VisionTransformer, PatchEmbedding


def test_patch_embedding():
    """Test patch embedding component."""
    # Test default parameters
    patch_embed = PatchEmbedding()
    assert patch_embed.img_size == 224
    assert patch_embed.patch_size == 16
    assert patch_embed.num_patches == (224 // 16) ** 2
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = patch_embed(x)
    expected_shape = (2, patch_embed.num_patches, 768)  # default embed_dim
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    
    # Test different image size
    patch_embed_small = PatchEmbedding(img_size=112, patch_size=16, embed_dim=256)
    x_small = torch.randn(2, 3, 112, 112)
    out_small = patch_embed_small(x_small)
    expected_shape_small = (2, (112 // 16) ** 2, 256)
    assert out_small.shape == expected_shape_small
    
    # Test invalid image size
    with pytest.raises(AssertionError):
        patch_embed(torch.randn(2, 3, 112, 112))  # Wrong size for 224x224 model


def test_vit_base():
    """Test ViT-Base model."""
    model = vit_base(num_classes=1000)
    
    # Check model architecture
    assert isinstance(model, VisionTransformer)
    assert model.embed_dim == 768
    assert len(model.blocks) == 12
    assert model.blocks[0].attn.num_heads == 12
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, 1000)
    
    # Test different number of classes
    model_small = vit_base(num_classes=10)
    out_small = model_small(x)
    assert out_small.shape == (4, 10)
    
    # Test different image size
    model_large_img = vit_base(num_classes=200, img_size=384)
    x_large = torch.randn(2, 3, 384, 384)
    out_large = model_large_img(x_large)
    assert out_large.shape == (2, 200)


def test_vit_large():
    """Test ViT-Large model."""
    model = vit_large(num_classes=1000)
    
    # Check model architecture
    assert isinstance(model, VisionTransformer)
    assert model.embed_dim == 1024
    assert len(model.blocks) == 24
    assert model.blocks[0].attn.num_heads == 16
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1000)
    
    # Test different configurations
    model_custom = vit_large(num_classes=200, drop_rate=0.1, attn_drop_rate=0.1)
    out_custom = model_custom(x)
    assert out_custom.shape == (2, 200)


def test_vit_parameter_counts():
    """Test that ViT models have expected parameter counts."""
    model_base = vit_base()
    model_large = vit_large()
    
    # Count parameters
    base_params = sum(p.numel() for p in model_base.parameters())
    large_params = sum(p.numel() for p in model_large.parameters())
    
    # ViT-Base should have ~86M parameters, ViT-Large should have ~304M
    assert 80e6 < base_params < 90e6, f"ViT-Base has {base_params/1e6:.1f}M params, expected ~86M"
    assert 300e6 < large_params < 310e6, f"ViT-Large has {large_params/1e6:.1f}M params, expected ~304M"
    
    # Large should have more parameters than base
    assert large_params > base_params


def test_vit_bias_configuration():
    """Test that ViT models follow ChillAdam's bias=False pattern."""
    model = vit_base()
    
    # Check that key linear layers have bias=False
    # Patch embedding projection
    assert model.patch_embed.proj.bias is None
    
    # Attention projections
    assert model.blocks[0].attn.proj.bias is None
    assert model.blocks[0].attn.qkv.bias is None  # Should be False by default
    
    # MLP layers
    assert model.blocks[0].mlp.fc1.bias is None
    assert model.blocks[0].mlp.fc2.bias is None
    
    # Classification head
    assert model.head.bias is None


def test_vit_dropout():
    """Test dropout functionality in ViT models."""
    # Test model with dropout
    model_with_dropout = vit_base(drop_rate=0.1, attn_drop_rate=0.1)
    
    # Set to train mode
    model_with_dropout.train()
    
    # Run multiple forward passes and check for variation (indicating dropout is working)
    x = torch.randn(4, 3, 224, 224)
    torch.manual_seed(42)
    out1 = model_with_dropout(x)
    torch.manual_seed(43)
    out2 = model_with_dropout(x)
    
    # Outputs should be different due to dropout
    assert not torch.allclose(out1, out2, atol=1e-6)
    
    # Test model without dropout
    model_no_dropout = vit_base(drop_rate=0.0, attn_drop_rate=0.0)
    model_no_dropout.eval()
    
    # Multiple forward passes should be identical in eval mode
    torch.manual_seed(42)
    out1_eval = model_no_dropout(x)
    torch.manual_seed(43)
    out2_eval = model_no_dropout(x)
    
    assert torch.allclose(out1_eval, out2_eval, atol=1e-6)


def test_vit_gradient_flow():
    """Test that gradients flow properly through ViT models."""
    model = vit_base(num_classes=10)
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    target = torch.randint(0, 10, (2,))
    
    # Forward pass
    out = model(x)
    loss = nn.CrossEntropyLoss()(out, target)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist for model parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient for parameter {name}"


def test_vit_different_patch_sizes():
    """Test ViT with different patch sizes."""
    # Test patch size 32
    model_p32 = VisionTransformer(img_size=224, patch_size=32, embed_dim=768, 
                                 depth=12, num_heads=12, num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    out_p32 = model_p32(x)
    assert out_p32.shape == (2, 1000)
    
    # Number of patches should be different
    assert model_p32.patch_embed.num_patches == (224 // 32) ** 2
    
    # Test patch size 8 (smaller patches, more computation)
    model_p8 = VisionTransformer(img_size=224, patch_size=8, embed_dim=384,
                                depth=6, num_heads=6, num_classes=1000)
    out_p8 = model_p8(x)
    assert out_p8.shape == (2, 1000)
    assert model_p8.patch_embed.num_patches == (224 // 8) ** 2


def test_vit_small_images():
    """Test ViT with smaller input images."""
    # Test with 112x112 images
    model_small = VisionTransformer(img_size=112, patch_size=16, embed_dim=384,
                                   depth=6, num_heads=6, num_classes=100)
    x_small = torch.randn(3, 3, 112, 112)
    out_small = model_small(x_small)
    assert out_small.shape == (3, 100)
    
    # Test with 64x64 images (like CIFAR)
    model_tiny = VisionTransformer(img_size=64, patch_size=8, embed_dim=256,
                                  depth=6, num_heads=8, num_classes=10)
    x_tiny = torch.randn(4, 3, 64, 64)
    out_tiny = model_tiny(x_tiny)
    assert out_tiny.shape == (4, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])