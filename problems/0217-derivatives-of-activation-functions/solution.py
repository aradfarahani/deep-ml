import math

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    # Sigmoid derivative: sigma'(x) = sigma(x) * (1 - sigma(x))
    sigmoid_val = 1 / (1 + math.exp(-x))
    sigmoid_deriv = sigmoid_val * (1 - sigmoid_val)
    
    # Tanh derivative: tanh'(x) = 1 - tanh^2(x)
    tanh_val = math.tanh(x)
    tanh_deriv = 1 - tanh_val ** 2
    
    # ReLU derivative: 1 if x > 0, else 0
    relu_deriv = 1.0 if x > 0 else 0.0
    
    return {
        'sigmoid': sigmoid_deriv,
        'tanh': tanh_deriv,
        'relu': relu_deriv
    }