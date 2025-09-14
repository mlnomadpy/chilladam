"""
Scheduler factory for creating different learning rate schedulers based on configuration.
"""

import torch.optim.lr_scheduler as lr_scheduler


def create_scheduler(scheduler_name, optimizer, **kwargs):
    """
    Create a learning rate scheduler based on the scheduler name and parameters.
    
    Arguments:
        scheduler_name: Name of the scheduler to create
        optimizer: PyTorch optimizer to schedule
        **kwargs: Scheduler-specific parameters
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Created scheduler or None if scheduler_name is "none"
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == "none" or not scheduler_name:
        return None
    
    elif scheduler_name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('t_max', 10),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    
    elif scheduler_name == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif scheduler_name == "exponential":
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    
    elif scheduler_name == "cosine_warm_restarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('t_0', 10),
            T_mult=kwargs.get('t_mult', 1),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}. "
                        f"Supported schedulers: cosine, step, exponential, cosine_warm_restarts, none")


def get_scheduler_info():
    """
    Get information about supported schedulers.
    
    Returns:
        dict: Dictionary with scheduler names as keys and their descriptions as values
    """
    return {
        "cosine": "Cosine Annealing Learning Rate scheduler",
        "step": "Step Learning Rate scheduler",
        "exponential": "Exponential Learning Rate scheduler", 
        "cosine_warm_restarts": "Cosine Annealing with Warm Restarts",
        "none": "No scheduler (constant learning rate)"
    }