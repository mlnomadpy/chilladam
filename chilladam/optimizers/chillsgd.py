"""
ChillSGD optimizer implementation.

A custom SGD optimizer that incorporates the "chill" mechanism from ChillAdam:
- Gradient normalization (dividing gradient by its norm)
- Adaptive learning rate based on inverse parameter norms
"""

import torch
from torch.optim.optimizer import Optimizer


class ChillSGD(Optimizer):
    """
    ChillSGD optimizer with gradient normalization and adaptive learning rate based on parameter norms.
    
    This is an SGD optimizer without momentum that incorporates the "chill" mechanism:
    - Gradients are normalized by their L2 norm
    - Learning rate is replaced by the inverse of the parameter norm
    
    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        min_lr: minimum learning rate (default: 1e-5)
        max_lr: maximum learning rate (default: 1.0)
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        lr: base learning rate for scheduler compatibility (default: max_lr)
            This parameter allows learning rate schedulers to work with ChillSGD
    """
    
    def __init__(self, params, min_lr=1e-5, max_lr=1.0, eps=1e-8, weight_decay=0, lr=None):
        if not 0.0 <= min_lr:
            raise ValueError(f"Invalid min_lr: {min_lr}")
        if not 0.0 <= max_lr:
            raise ValueError(f"Invalid max_lr: {max_lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        # Set lr to the max_lr if not provided (for scheduler compatibility)
        if lr is None:
            lr = max_lr
            
        defaults = dict(min_lr=min_lr, max_lr=max_lr, eps=eps, weight_decay=weight_decay, lr=lr)
        super(ChillSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            min_lr, max_lr, eps, weight_decay = group['min_lr'], group['max_lr'], group['eps'], group['weight_decay']
            scheduler_lr = group['lr']  # Learning rate from scheduler (or initial lr)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('ChillSGD does not support sparse gradients')

                # Apply weight decay to gradient if specified
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Chill mechanism 1:
                grad_normalized = grad 

                # Chill mechanism 2: Use inverse of parameter norm as learning rate
                param_norm = p.norm(p=2).clamp(min=eps)
                adaptive_lr = 1.0 / param_norm
                adaptive_lr = adaptive_lr.clamp(min=min_lr, max=max_lr)
                
                # Scale the adaptive learning rate by the scheduler's learning rate
                lr = adaptive_lr * (scheduler_lr / max_lr)  # Normalize by max_lr to maintain scaling

                # Store the learning rate for debugging/monitoring
                state = self.state[p]
                state["lr"] = lr.item()

                # Perform SGD update: p = p - lr * normalized_grad
                p.add_(grad_normalized, alpha=-lr)

        return loss
