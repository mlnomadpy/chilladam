"""
Vision Transformer (ViT) implementation for ChillAdam library.

This module provides Vision Transformer models following the original ViT paper
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
with adaptations to match ChillAdam's design patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """
    Convert image patches into embeddings.
    
    Args:
        img_size: Input image size (default: 224)
        patch_size: Size of each patch (default: 16)
        in_channels: Number of input channels (default: 3)
        embed_dim: Embedding dimension (default: 768)
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding using convolution
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input image size ({H}x{W}) doesn't match model ({self.img_size}x{self.img_size})"
        
        # Apply patch embedding: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        # Flatten: (B, embed_dim, num_patches_h, num_patches_w) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        # Transpose: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to add bias to q, k, v projections (default: False following ChillAdam patterns)
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        # Generate q, k, v: (B, N, C) -> (B, N, 3*C) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation: (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention: (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for transformer blocks.
    
    Args:
        in_features: Input feature dimension
        hidden_features: Hidden layer dimension (if None, defaults to 4 * in_features)
        out_features: Output feature dimension (if None, defaults to in_features)
        drop: Dropout rate
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or 4 * in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and MLP.
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to input dimension
        qkv_bias: Whether to add bias to q, k, v projections
        drop: Dropout rate
        attn_drop: Attention dropout rate
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                     attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
    def forward(self, x):
        # Pre-norm residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer model.
    
    Args:
        img_size: Input image size
        patch_size: Size of each patch
        in_channels: Number of input channels
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
        qkv_bias: Whether to add bias to q, k, v projections
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False,
                 drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, 
                                        in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes, bias=False) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize positional embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
        def _init_linear(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        self.apply(_init_linear)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        
        # Add class token: (B, num_patches, embed_dim) -> (B, num_patches + 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final norm and classification
        x = self.norm(x)
        
        # Use class token for classification
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        
        return x


def vit_base(num_classes=1000, img_size=224, patch_size=16, drop_rate=0.0, attn_drop_rate=0.0):
    """
    ViT-Base model (ViT-B/16).
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size  
        patch_size: Size of each patch
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """
    return VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=num_classes,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        qkv_bias=False, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate
    )


def vit_large(num_classes=1000, img_size=224, patch_size=16, drop_rate=0.0, attn_drop_rate=0.0):
    """
    ViT-Large model (ViT-L/16).
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        patch_size: Size of each patch  
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """
    return VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=num_classes,
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0,
        qkv_bias=False, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate
    )