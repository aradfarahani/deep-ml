import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    M = A.astype(float).copy()
    m, n = M.shape
    
    rank = 0
    pivot_row = 0
    
    for col in range(n):
        if pivot_row >= m:
            break
        
        # Find the row with the maximum absolute value in this column
        max_idx = pivot_row
        max_val = abs(M[pivot_row, col])
        for i in range(pivot_row + 1, m):
            if abs(M[i, col]) > max_val:
                max_val = abs(M[i, col])
                max_idx = i
        
        # If the maximum value is below tolerance, skip this column
        if max_val < tol:
            continue
        
        # Swap rows
        M[[pivot_row, max_idx]] = M[[max_idx, pivot_row]]
        
        # Eliminate entries below the pivot
        for i in range(pivot_row + 1, m):
            if abs(M[i, col]) > tol:
                factor = M[i, col] / M[pivot_row, col]
                M[i] = M[i] - factor * M[pivot_row]
        
        rank += 1
        pivot_row += 1
    
    return rank