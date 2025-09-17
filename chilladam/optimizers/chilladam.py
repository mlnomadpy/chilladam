"""
ChillAdam optimizer implementation.

A custom optimizer that adapts learning rates based on parameter norms.
"""

import torch
from torch.optim.optimizer import Optimizer


class ChillAdam(Optimizer):
    """
    ChillAdam optimizer with adaptive learning rate based on parameter norms.
    
    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        min_lr: minimum learning rate (default: 1e-5)
        max_lr: maximum learning rate (default: 1.0)
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        betas: coefficients used for computing running averages of gradient
               and its square (default: (0.9, 0.999))
        weight_decay: weight decay (L2 penalty) (default: 0)
        lr: base learning rate for scheduler compatibility (default: max_lr)
            This parameter allows learning rate schedulers to work with ChillAdam
    """
    
    def __init__(self, params, min_lr=1e-5, max_lr=1.0, eps=1e-8, betas=(0.9, 0.999), weight_decay=0, lr=None):
        if not 0.0 <= min_lr:
            raise ValueError(f"Invalid min_lr: {min_lr}")
        if not 0.0 <= max_lr:
            raise ValueError(f"Invalid max_lr: {max_lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        # Set lr to the max_lr if not provided (for scheduler compatibility)
        if lr is None:
            lr = max_lr
        
        defaults = dict(min_lr=min_lr, max_lr=max_lr, eps=eps, betas=betas, weight_decay=weight_decay, lr=lr)
        super(ChillAdam, self).__init__(params, defaults)

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
            min_lr, max_lr, eps, betas, weight_decay = group['min_lr'], group['max_lr'], group['eps'], group['betas'], group['weight_decay']
            scheduler_lr = group['lr']  # Learning rate from scheduler (or initial lr)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('ChillAdam does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = betas
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                grad_normalized = grad 

                if weight_decay != 0:
                    grad_normalized = grad_normalized.add(p, alpha=weight_decay)

                exp_avg.mul_(beta1).add_(grad_normalized, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_normalized, grad_normalized, value=1 - beta2)

                param_norm = p.norm(p=2).clamp(min=eps)
                adaptive_lr = 1.0 / param_norm
                adaptive_lr = adaptive_lr.clamp(min=min_lr, max=max_lr)
                
                # Scale the adaptive learning rate by the scheduler's learning rate
                # This allows schedulers to influence ChillAdam's learning rate
                lr = adaptive_lr * (scheduler_lr / max_lr)  # Normalize by max_lr to maintain scaling
                self.state[p]["lr"] = lr.item()

                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
