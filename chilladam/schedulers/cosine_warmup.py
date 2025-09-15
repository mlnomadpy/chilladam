"""
Cosine Annealing with Linear Warmup Scheduler.

This scheduler implements:
1. Linear warmup from 0 to the initial learning rate over warmup_epochs
2. Cosine annealing from initial LR to eta_min over the remaining epochs
"""

import math
import torch.optim.lr_scheduler as lr_scheduler


class CosineAnnealingWarmupScheduler(lr_scheduler._LRScheduler):
    """
    Cosine Annealing with Linear Warmup Scheduler.
    
    This scheduler combines linear warmup with cosine annealing:
    - Phase 1: Linear warmup from 0 to base_lr over warmup_epochs
    - Phase 2: Cosine annealing from base_lr to eta_min over (total_epochs - warmup_epochs)
    
    Arguments:
        optimizer: Wrapped optimizer.
        total_epochs: Total number of training epochs.
        warmup_epochs: Number of epochs for linear warmup phase.
        eta_min: Minimum learning rate (default: 0).
        last_epoch: The index of last epoch (default: -1).
    """
    
    def __init__(self, optimizer, total_epochs, warmup_epochs=0, eta_min=0, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        
        # Store original learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
        
        # If we have warmup and this is the first initialization, set LR to 0
        if warmup_epochs > 0 and last_epoch == -1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0
    
    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.")
        
        epoch = self.last_epoch
        
        if epoch < self.warmup_epochs and self.warmup_epochs > 0:
            # Linear warmup phase: grow linearly from 0 to base_lr
            # epoch -1 (initial): lr = 0 (handled in __init__)
            # epoch 0: lr = base_lr / warmup_epochs
            # epoch warmup_epochs-1: lr = base_lr
            warmup_factor = (epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cosine_epochs = self.total_epochs - self.warmup_epochs
            cosine_epoch = epoch - self.warmup_epochs
            
            if cosine_epochs <= 0:
                return [self.eta_min for _ in self.base_lrs]
            
            # Cosine annealing formula
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * cosine_epoch / cosine_epochs)) / 2 
                   for base_lr in self.base_lrs]
    
    def _get_closed_form_lr(self):
        """Return the learning rate using closed form computation."""
        return self.get_lr()