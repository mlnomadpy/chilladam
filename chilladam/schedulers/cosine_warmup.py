"""
Cosine Annealing with Linear Warmup Scheduler with Restart Support and Optional Linear Decay.

This scheduler implements:
1. Linear warmup from 0 to the initial learning rate over warmup_epochs
2. Cosine annealing from initial LR to eta_min over the remaining epochs
3. Optional linear decay phase after cosine annealing
4. Restart behavior - cycles repeat after total_epochs (with optional linear decay)
"""

import math
import torch.optim.lr_scheduler as lr_scheduler


class CosineAnnealingWarmupScheduler(lr_scheduler._LRScheduler):
    """
    Cosine Annealing with Linear Warmup Scheduler with Restart Support and Optional Linear Decay.
    
    This scheduler combines linear warmup with cosine annealing and supports restart cycles:
    - Phase 1: Linear warmup from 0 to base_lr over warmup_epochs
    - Phase 2: Cosine annealing from base_lr to eta_min over cosine_epochs
    - Phase 3 (optional): Linear decay from eta_min to final_lr over linear_decay_epochs
    - Restart: After total_epochs, the cycle restarts from Phase 1
    
    Arguments:
        optimizer: Wrapped optimizer.
        total_epochs: Total number of training epochs per cycle.
        warmup_epochs: Number of epochs for linear warmup phase.
        eta_min: Minimum learning rate after cosine annealing (default: 0).
        linear_decay_epochs: Number of epochs for optional linear decay phase (default: 0).
        final_lr: Final learning rate after linear decay phase (default: same as eta_min).
        restart: Whether to restart the cycle after total_epochs (default: True).
        last_epoch: The index of last epoch (default: -1).
    """
    
    def __init__(self, optimizer, total_epochs, warmup_epochs=0, eta_min=0, 
                 linear_decay_epochs=0, final_lr=None, restart=True, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.linear_decay_epochs = linear_decay_epochs
        self.eta_min = eta_min
        self.final_lr = final_lr if final_lr is not None else eta_min
        self.restart = restart
        
        # Calculate cosine epochs (excluding warmup and linear decay)
        self.cosine_epochs = total_epochs - warmup_epochs - linear_decay_epochs
        
        if self.cosine_epochs < 0:
            raise ValueError(f"total_epochs ({total_epochs}) must be >= warmup_epochs ({warmup_epochs}) + linear_decay_epochs ({linear_decay_epochs})")
        
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
        
        # Handle restart behavior
        if self.restart and epoch >= 0:
            cycle_epoch = epoch % self.total_epochs
        else:
            cycle_epoch = epoch
            # For no restart, just use the epoch as-is, but we might go beyond total_epochs
        
        # Phase 1: Linear warmup
        if cycle_epoch < self.warmup_epochs and self.warmup_epochs > 0:
            # Linear warmup phase: grow linearly from 0 to base_lr
            # Special handling: for restart behavior, the very first epoch of each cycle
            # should start from 0, and then grow to base_lr over warmup_epochs steps
            if cycle_epoch == 0 and self.restart and epoch >= 0:
                # At the start of each cycle (including restarts), begin from 0
                return [0.0 for _ in self.base_lrs]
            elif epoch == -1:  # Initial case
                return [0.0 for _ in self.base_lrs]
            else:
                # Standard warmup: cycle_epoch + 1 goes from 1 to warmup_epochs
                warmup_factor = (cycle_epoch + 1) / self.warmup_epochs
                return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # Phase 2: Cosine annealing
        elif cycle_epoch < self.warmup_epochs + self.cosine_epochs or not self.restart:
            cosine_epoch = cycle_epoch - self.warmup_epochs
            
            if self.cosine_epochs <= 0:
                return [self.eta_min for _ in self.base_lrs]
            
            # For no restart, continue cosine annealing indefinitely
            if not self.restart and cosine_epoch >= self.cosine_epochs:
                # Continue the cosine pattern beyond the original total_epochs
                # This provides a continuous cosine schedule
                extended_cosine_epoch = cosine_epoch
                extended_cosine_epochs = max(self.cosine_epochs, extended_cosine_epoch + 1)
                cosine_factor = (1 + math.cos(math.pi * extended_cosine_epoch / extended_cosine_epochs)) / 2
            else:
                # Standard cosine annealing formula
                cosine_factor = (1 + math.cos(math.pi * cosine_epoch / self.cosine_epochs)) / 2
            
            return [self.eta_min + (base_lr - self.eta_min) * cosine_factor 
                   for base_lr in self.base_lrs]
        
        # Phase 3: Optional linear decay (only applies with restart=True)
        elif self.linear_decay_epochs > 0 and self.restart:
            decay_epoch = cycle_epoch - self.warmup_epochs - self.cosine_epochs
            
            # Linear decay from eta_min to final_lr
            if decay_epoch >= self.linear_decay_epochs:
                # Past the decay phase, return final_lr
                return [self.final_lr for _ in self.base_lrs]
            else:
                # Linear interpolation: decay_epoch goes from 0 to linear_decay_epochs-1
                # We want to map this to eta_min -> final_lr
                decay_factor = (decay_epoch + 1) / self.linear_decay_epochs
                return [self.eta_min + (self.final_lr - self.eta_min) * decay_factor 
                       for _ in self.base_lrs]
        
        # Default: return final_lr if in linear decay, otherwise eta_min
        else:
            if self.linear_decay_epochs > 0:
                return [self.final_lr for _ in self.base_lrs]
            else:
                return [self.eta_min for _ in self.base_lrs]
    
    def _get_closed_form_lr(self):
        """Return the learning rate using closed form computation."""
        return self.get_lr()