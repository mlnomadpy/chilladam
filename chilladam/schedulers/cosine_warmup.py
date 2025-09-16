"""
Cosine Annealing with Linear Warmup Scheduler.

This scheduler implements:
1. Linear warmup from 0 to the initial learning rate over warmup_epochs
2. Cosine annealing from initial LR to eta_min over the cosine phase
3. Linear decay from eta_min to final_lr for remaining epochs (if any)
"""

import math
import torch.optim.lr_scheduler as lr_scheduler


class CosineAnnealingWarmupScheduler(lr_scheduler._LRScheduler):
    """
    Cosine Annealing with Linear Warmup Scheduler.
    
    This scheduler combines three phases:
    - Phase 1: Linear warmup from 0 to base_lr over warmup_epochs
    - Phase 2: Cosine annealing from base_lr to eta_min over cosine_epochs
    - Phase 3: Linear decay from eta_min to final_lr over remaining epochs (optional)
    
    Arguments:
        optimizer: Wrapped optimizer.
        total_epochs: Total number of training epochs.
        warmup_epochs: Number of epochs for linear warmup phase.
        cosine_epochs: Number of epochs for cosine annealing phase. If None, uses (total_epochs - warmup_epochs).
        eta_min: Minimum learning rate at end of cosine phase (default: 0).
        final_lr: Final learning rate for linear decay phase. If None, uses eta_min (no decay phase).
        last_epoch: The index of last epoch (default: -1).
    """
    
    def __init__(self, optimizer, total_epochs, warmup_epochs=0, cosine_epochs=None, eta_min=0, final_lr=None, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.final_lr = final_lr if final_lr is not None else eta_min
        
        # If cosine_epochs not specified, use remaining epochs after warmup
        if cosine_epochs is None:
            self.cosine_epochs = total_epochs - warmup_epochs
        else:
            self.cosine_epochs = cosine_epochs
        
        # Calculate when linear decay phase starts
        self.decay_start_epoch = warmup_epochs + self.cosine_epochs
        
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
            # Phase 1: Linear warmup phase: grow linearly from 0 to base_lr
            # epoch -1 (initial): lr = 0 (handled in __init__)
            # epoch 0: lr = base_lr / warmup_epochs
            # epoch warmup_epochs-1: lr = base_lr
            warmup_factor = (epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        elif epoch < self.decay_start_epoch:
            # Phase 2: Cosine annealing phase
            cosine_epoch = epoch - self.warmup_epochs
            
            if self.cosine_epochs <= 0:
                return [self.eta_min for _ in self.base_lrs]
            
            # Cosine annealing formula
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * cosine_epoch / self.cosine_epochs)) / 2 
                   for base_lr in self.base_lrs]
        else:
            # Phase 3: Linear decay phase (from eta_min to final_lr)
            if self.final_lr == self.eta_min:
                # No decay phase - maintain eta_min
                return [self.eta_min for _ in self.base_lrs]
            
            # Calculate how many epochs are left for decay
            decay_epochs = self.total_epochs - self.decay_start_epoch
            if decay_epochs <= 0:
                return [self.final_lr for _ in self.base_lrs]
            
            # Current position in decay phase
            decay_epoch = epoch - self.decay_start_epoch
            
            # Clamp to avoid going beyond total_epochs
            if decay_epoch >= decay_epochs:
                return [self.final_lr for _ in self.base_lrs]
            
            # Linear interpolation from eta_min to final_lr
            decay_factor = decay_epoch / decay_epochs
            return [self.eta_min + (self.final_lr - self.eta_min) * decay_factor 
                   for _ in self.base_lrs]
    
    def _get_closed_form_lr(self):
        """Return the learning rate using closed form computation."""
        return self.get_lr()