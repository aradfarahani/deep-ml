import math

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
    """
    Compute the derivative of cross-entropy loss with respect to logits.
    
    Args:
        logits: Raw model outputs (before softmax)
        target: Index of the true class
        
    Returns:
        Gradient vector: dL/dz_i = softmax(z)_i - y_i
    """
    # Compute softmax probabilities
    max_z = max(logits)
    exp_z = [math.exp(z - max_z) for z in logits]
    sum_exp = sum(exp_z)
    probs = [e / sum_exp for e in exp_z]
    
    # Gradient = softmax - one_hot(target)
    gradients = probs.copy()
    gradients[target] -= 1.0
    
    return gradients