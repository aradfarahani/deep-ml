import math

def warmup_cosine_schedule(T: int, W: int, lr_max: float, lr_min: float) -> list[float]:
    """
    Compute learning rate schedule with linear warmup and cosine decay.
    
    Args:
        T: Total number of training steps
        W: Number of warmup steps
        lr_max: Maximum learning rate (reached after warmup)
        lr_min: Minimum learning rate (reached at end of training)
    
    Returns:
        List of learning rates for each step
    """
    lr_schedule = []
    
    for step in range(T):
        if step < W:
            # Linear warmup phase
            if W == 0:
                lr = lr_max
            else:
                lr = (step / W) * lr_max
        else:
            # Cosine decay phase
            decay_steps = T - W
            if decay_steps == 0:
                lr = lr_max
            else:
                progress = (step - W) / decay_steps
                lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))
        
        lr_schedule.append(lr)
    
    return lr_schedule