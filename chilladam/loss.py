"""
Loss function utilities including L1 (Lasso) regularization.

This module provides utilities for adding regularization terms to loss functions,
following the pattern of computing regularization separately and adding it to the base loss.
"""

import torch


def add_l1_regularization(loss, model, l1_lambda):
    """
    Add L1 (Lasso) regularization to a loss value.
    
    Args:
        loss (torch.Tensor): The base loss tensor
        model (torch.nn.Module): The model whose parameters should be regularized
        l1_lambda (float): L1 regularization strength. If 0, no regularization is added.
        
    Returns:
        torch.Tensor: The loss with L1 regularization added
        
    Example:
        >>> criterion = nn.CrossEntropyLoss()
        >>> base_loss = criterion(outputs, targets)
        >>> regularized_loss = add_l1_regularization(base_loss, model, l1_lambda=0.01)
        >>> regularized_loss.backward()
    """
    if l1_lambda <= 0:
        return loss
    
    # L1 penalty - sum of absolute values of all parameters
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return loss + l1_lambda * l1_norm


class L1RegularizedLoss:
    """
    A wrapper class that adds L1 (Lasso) regularization to any loss function.
    
    This class wraps a base loss function and automatically adds L1 regularization
    to the computed loss when called.
    
    Args:
        base_loss_fn (callable): The base loss function (e.g., nn.CrossEntropyLoss())
        l1_lambda (float): L1 regularization strength. Default: 0.0 (no regularization)
        
    Example:
        >>> base_criterion = nn.CrossEntropyLoss()
        >>> criterion = L1RegularizedLoss(base_criterion, l1_lambda=0.01)
        >>> loss = criterion(outputs, targets, model)
        >>> loss.backward()
    """
    
    def __init__(self, base_loss_fn, l1_lambda=0.0):
        if l1_lambda < 0:
            raise ValueError(f"l1_lambda must be non-negative, got {l1_lambda}")
        
        self.base_loss_fn = base_loss_fn
        self.l1_lambda = l1_lambda
    
    def __call__(self, predictions, targets, model):
        """
        Compute the loss with L1 regularization.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            model (torch.nn.Module): The model whose parameters should be regularized
            
        Returns:
            torch.Tensor: The regularized loss
        """
        base_loss = self.base_loss_fn(predictions, targets)
        return add_l1_regularization(base_loss, model, self.l1_lambda)
    
    def set_l1_lambda(self, l1_lambda):
        """Update the L1 regularization strength."""
        if l1_lambda < 0:
            raise ValueError(f"l1_lambda must be non-negative, got {l1_lambda}")
        self.l1_lambda = l1_lambda
    
    def get_l1_lambda(self):
        """Get the current L1 regularization strength."""
        return self.l1_lambda