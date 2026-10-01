import numpy as np
import math

def rmsprop_update(params: list[float], grads: list[float], cache: list[float], 
                   lr: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8) -> tuple[list[float], list[float]]:
    """
    Perform RMSProp optimization update.
    
    Args:
        params: List of parameter values
        grads: List of gradients for each parameter
        cache: List of cache values (moving average of squared gradients)
        lr: Learning rate
        beta: Decay rate for moving average (typically 0.9)
        epsilon: Small constant for numerical stability
    
    Returns:
        Tuple of (updated_params, updated_cache)
    """
    updated_params = []
    updated_cache = []
    
    for param, grad, v in zip(params, grads, cache):
        # Update cache (moving average of squared gradients)
        v_new = beta * v + (1 - beta) * (grad ** 2)
        
        # Update parameter
        param_new = param - lr * grad / (math.sqrt(v_new) + epsilon)
        
        updated_params.append(param_new)
        updated_cache.append(v_new)
    
    return updated_params, updated_cache