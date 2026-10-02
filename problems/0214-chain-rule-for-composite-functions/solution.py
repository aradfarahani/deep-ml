import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    """
    # Define functions and their derivatives
    func_map = {
        'square': (lambda t: t**2, lambda t: 2*t),
        'sin': (lambda t: np.sin(t), lambda t: np.cos(t)),
        'exp': (lambda t: np.exp(t), lambda t: np.exp(t)),
        'log': (lambda t: np.log(t), lambda t: 1/t)
    }
    
    # Compute nested function values
    values = [x]
    for func_name in reversed(functions):
        func, _ = func_map[func_name]
        values.append(func(values[-1]))
    
    # Apply chain rule: multiply all derivatives
    gradient = 1.0
    for i, func_name in enumerate(reversed(functions)):
        _, deriv = func_map[func_name]
        gradient *= deriv(values[i])
    
    return float(gradient)