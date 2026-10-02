import numpy as np

def temperature_decay(
    schedule_type: str,
    initial_temp: float,
    current_step: int,
    total_steps: int,
    final_temp: float = 0.01,
    decay_rate: float = 0.95
) -> float:
    """
    Compute temperature at current training step.
    
    Args:
        schedule_type: 'linear', 'exponential', 'cosine', or 'constant'
        initial_temp: Starting temperature
        current_step: Current training step
        total_steps: Total number of steps
        final_temp: Minimum temperature
        decay_rate: Decay rate for exponential
    
    Returns:
        Temperature value at current step
    """
    if schedule_type == 'constant':
        return initial_temp
    
    elif schedule_type == 'linear':
        progress = current_step / total_steps
        temp = initial_temp - (initial_temp - final_temp) * progress
        return max(temp, final_temp)
    
    elif schedule_type == 'exponential':
        temp = initial_temp * (decay_rate ** current_step)
        return max(temp, final_temp)
    
    elif schedule_type == 'cosine':
        progress = current_step / total_steps
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        temp = final_temp + (initial_temp - final_temp) * cosine_decay
        return temp
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")