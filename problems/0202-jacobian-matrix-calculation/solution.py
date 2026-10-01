import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Jacobian matrix of a vector-valued function using numerical differentiation.
    
    Args:
        f: A function that takes a list of floats and returns a list of floats
        x: Point at which to evaluate the Jacobian (list of n values)
        h: Step size for numerical differentiation
    
    Returns:
        Jacobian matrix as list of lists (m x n matrix)
    """
    x = np.array(x, dtype=float)
    n = len(x)
    
    # Evaluate function at x to get output dimension
    f_x = np.array(f(x.tolist()))
    m = len(f_x)
    
    # Initialize Jacobian matrix
    J = np.zeros((m, n))
    
    # Compute partial derivatives using finite differences
    for j in range(n):
        # Perturb x in the j-th direction
        x_plus = x.copy()
        x_plus[j] += h
        
        # Compute finite difference approximation
        f_plus = np.array(f(x_plus.tolist()))
        J[:, j] = (f_plus - f_x) / h
    
    return J.tolist()