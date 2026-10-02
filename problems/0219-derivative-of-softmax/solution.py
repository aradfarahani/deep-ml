import math

def softmax_derivative(x: list[float]) -> list[list[float]]:
    """
    Compute the Jacobian matrix of the softmax function.
    
    Args:
        x: Input vector
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    # First compute softmax
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    s = [e / sum_exp for e in exp_x]
    
    # Compute Jacobian
    n = len(s)
    jacobian = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                jacobian[i][j] = s[i] * (1 - s[i])
            else:
                jacobian[i][j] = -s[i] * s[j]
    
    return jacobian