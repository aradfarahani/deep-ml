import math

def clip_gradients_by_global_norm(gradients: list[list[float]], max_norm: float) -> list[list[float]]:
    """
    Clip gradients by global norm.
    
    Args:
        gradients: List of gradient arrays (each can be multi-dimensional, flattened to 1D lists)
        max_norm: Maximum allowed global norm
    
    Returns:
        List of clipped gradient arrays
    """
    # Compute global norm (L2 norm of all gradients combined)
    global_norm = 0.0
    for grad_array in gradients:
        for val in grad_array:
            global_norm += val ** 2
    global_norm = math.sqrt(global_norm)
    
    # Compute scaling factor
    clip_coefficient = max_norm / (global_norm + 1e-6)  # Add small epsilon to avoid division by zero
    
    # If global norm exceeds max_norm, scale all gradients
    if clip_coefficient < 1.0:
        clipped_gradients = []
        for grad_array in gradients:
            clipped_array = [val * clip_coefficient for val in grad_array]
            clipped_gradients.append(clipped_array)
        return clipped_gradients
    else:
        # No clipping needed
        return gradients