import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Args:
        A: An m x n matrix represented as list of lists
    
    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    A_np = np.array(A, dtype=float)
    m, n = A_np.shape
    
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        # Start with the j-th column of A
        v = A_np[:, j].copy()
        
        # Subtract projections onto previous Q columns (Gram-Schmidt)
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A_np[:, j])
            v = v - R[i, j] * Q[:, i]
        
        # Normalize
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q.tolist(), R.tolist()