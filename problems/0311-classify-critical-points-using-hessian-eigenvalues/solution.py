import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    hessian = np.array(hessian, dtype=float)
    eigenvalues = np.linalg.eigvalsh(hessian)

    positive = np.sum(eigenvalues > tol)
    negative = np.sum(eigenvalues < -tol)
    zero = np.sum(np.abs(eigenvalues) <= tol)

    n = len(eigenvalues)

    if zero > 0:
        return None
    elif positive == n:
        return -1
    elif negative == n:
        return 1
    else:
        return 0