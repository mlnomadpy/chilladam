"""
Optimizer factory for creating different optimizers based on configuration.
"""

import torch
import torch.optim as optim
from .chilladam import ChillAdam
from .chillsgd import ChillSGD


def create_optimizer(optimizer_name, model_parameters, **kwargs):
    """
    Create an optimizer based on the optimizer name and parameters.
    
    Arguments:
        optimizer_name: Name of the optimizer to create
        model_parameters: Model parameters to optimize
        **kwargs: Optimizer-specific parameters
        
    Returns:
        torch.optim.Optimizer: Created optimizer
        
    Note:
        L1 regularization (l1_lambda) is handled by the loss function, not the optimizer.
        If l1_lambda is provided, it will be ignored with a warning.
    """
    optimizer_name = optimizer_name.lower()
    
    # Check for l1_lambda and warn if provided
    if 'l1_lambda' in kwargs and kwargs['l1_lambda'] > 0:
        print(f"Warning: l1_lambda={kwargs['l1_lambda']} was provided but will be ignored.")
        print("L1 regularization is now handled by the loss function, not the optimizer.")
        print("Please use L1RegularizedLoss in your training code to enable L1 regularization.")
        # Remove l1_lambda from kwargs to avoid errors
        kwargs = {k: v for k, v in kwargs.items() if k != 'l1_lambda'}
    
    if optimizer_name == "chilladam":
        return ChillAdam(
            model_parameters,
            min_lr=kwargs.get('min_lr', 1e-5),
            max_lr=kwargs.get('max_lr', 1.0),
            eps=kwargs.get('eps', 1e-8),
            betas=kwargs.get('betas', (0.9, 0.999)),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    
    elif optimizer_name == "chillsgd":
        return ChillSGD(
            model_parameters,
            min_lr=kwargs.get('min_lr', 1e-5),
            max_lr=kwargs.get('max_lr', 1.0),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    
    elif optimizer_name == "adam":
        return optim.Adam(
            model_parameters,
            lr=kwargs.get('lr', 1e-3),
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    
    elif optimizer_name == "adamw":
        return optim.AdamW(
            model_parameters,
            lr=kwargs.get('lr', 1e-3),
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 1e-2)
        )
    
    elif optimizer_name == "sgd":
        return optim.SGD(
            model_parameters,
            lr=kwargs.get('lr', 1e-3),
            momentum=kwargs.get('momentum', 0),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(
            model_parameters,
            lr=kwargs.get('lr', 1e-3),
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0),
            momentum=kwargs.get('momentum', 0)
        )
    
    elif optimizer_name == "adamax":
        return optim.Adamax(
            model_parameters,
            lr=kwargs.get('lr', 2e-3),
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    
    elif optimizer_name == "nadam":
        return optim.NAdam(
            model_parameters,
            lr=kwargs.get('lr', 2e-3),
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    
    elif optimizer_name == "radam":
        return optim.RAdam(
            model_parameters,
            lr=kwargs.get('lr', 1e-3),
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. "
                        f"Supported optimizers: chilladam, chillsgd, adam, adamw, sgd, rmsprop, adamax, nadam, radam")


def get_optimizer_info():
    """
    Get information about supported optimizers.
    
    Returns:
        dict: Dictionary with optimizer names as keys and their descriptions as values
    """
    return {
        "chilladam": "Custom ChillAdam optimizer with adaptive learning rates",
        "chillsgd": "Custom ChillSGD optimizer with gradient normalization and adaptive learning rates (SGD without momentum)",
        "adam": "Adam optimizer",
        "adamw": "AdamW optimizer with decoupled weight decay",
        "sgd": "Stochastic Gradient Descent",
        "rmsprop": "RMSprop optimizer",
        "adamax": "Adamax optimizer (variant of Adam)",
        "nadam": "NAdam optimizer (Adam with Nesterov momentum)",
        "radam": "RAdam optimizer (Rectified Adam)"
    }