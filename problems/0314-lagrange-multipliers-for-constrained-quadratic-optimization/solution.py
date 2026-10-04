import numpy as np

def lagrange_optimize(Q: np.ndarray, c: np.ndarray, a: np.ndarray, b: float) -> dict:
    """
    Solve constrained quadratic optimization using Lagrange multipliers.
    
    Minimize: f(x) = (1/2) x^T Q x + c^T x
    Subject to: a^T x = b
    
    Args:
        Q: 2x2 symmetric positive definite matrix
        c: 2-element vector (linear coefficients)
        a: 2-element vector (constraint coefficients)
        b: scalar (constraint value)
    
    Returns:
        Dictionary with 'x', 'lambda', and 'objective' keys
    """
    Q = np.array(Q, dtype=float)
    c = np.array(c, dtype=float)
    a = np.array(a, dtype=float)
    
    # Build the KKT system matrix
    # [Q   -a] [x]   [-c]
    # [a^T  0] [lam] = [b ]
    KKT = np.zeros((3, 3))
    KKT[:2, :2] = Q
    KKT[:2, 2] = -a
    KKT[2, :2] = a
    
    # Build the right-hand side
    rhs = np.zeros(3)
    rhs[:2] = -c
    rhs[2] = b
    
    # Solve the KKT system
    solution = np.linalg.solve(KKT, rhs)
    x_opt = solution[:2]
    lambda_opt = solution[2]
    
    # Compute objective function value at optimum
    objective = 0.5 * x_opt @ Q @ x_opt + c @ x_opt
    
    return {
        'x': [round(float(v), 4) for v in x_opt],
        'lambda': round(float(lambda_opt), 4),
        'objective': round(float(objective), 4)
    }