from typing import Callable
import math

def newtons_method_optimization(
    gradient_func: Callable[[list[float]], list[float]],
    hessian_func: Callable[[list[float]], list[list[float]]],
    x0: list[float],
    tol: float = 1e-6,
    max_iter: int = 100
) -> list[float]:
    """
    Find the minimum of a function using Newton's method.
    
    Args:
        gradient_func: Function that returns gradient at a point
        hessian_func: Function that returns Hessian matrix at a point
        x0: Initial guess
        tol: Convergence tolerance for gradient norm
        max_iter: Maximum iterations
        
    Returns:
        The point that minimizes the function
    """
    x = x0.copy()
    n = len(x)
    
    for _ in range(max_iter):
        grad = gradient_func(x)
        hess = hessian_func(x)
        
        # Check convergence
        grad_norm = math.sqrt(sum(g**2 for g in grad))
        if grad_norm < tol:
            break
        
        # Compute Newton step: delta = -H^{-1} * grad
        if n == 1:
            delta = [-grad[0] / hess[0][0]]
        elif n == 2:
            # 2x2 matrix inverse
            a, b = hess[0][0], hess[0][1]
            c, d = hess[1][0], hess[1][1]
            det = a * d - b * c
            # H^{-1} * grad
            delta = [
                -(d * grad[0] - b * grad[1]) / det,
                -(-c * grad[0] + a * grad[1]) / det
            ]
        else:
            # For larger dimensions, use Gaussian elimination
            delta = solve_linear_system(hess, [-g for g in grad])
        
        # Update x
        x = [x[i] + delta[i] for i in range(n)]
    
    return x

def solve_linear_system(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    # Create augmented matrix
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = max(range(i, n), key=lambda r: abs(aug[r][i]))
        aug[i], aug[max_row] = aug[max_row], aug[i]
        
        for j in range(i + 1, n):
            factor = aug[j][i] / aug[i][i]
            for k in range(i, n + 1):
                aug[j][k] -= factor * aug[i][k]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))) / aug[i][i]
    
    return x