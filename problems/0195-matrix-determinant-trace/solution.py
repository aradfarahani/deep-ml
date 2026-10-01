import numpy as np

def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
    """
    Compute the determinant and trace of a square matrix.
    
    Args:
        matrix: A square matrix (n x n) represented as list of lists
    
    Returns:
        Tuple of (determinant, trace)
    """
    n = len(matrix)
    
    # Compute trace (sum of diagonal elements)
    trace = sum(matrix[i][i] for i in range(n))
    
    # Compute determinant using recursive expansion
    def determinant(mat):
        n = len(mat)
        
        # Base case: 1x1 matrix
        if n == 1:
            return mat[0][0]
        
        # Base case: 2x2 matrix
        if n == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        
        # Recursive case: use cofactor expansion along first row
        det = 0
        for j in range(n):
            # Create minor matrix (remove row 0 and column j)
            minor = [[mat[i][k] for k in range(n) if k != j] 
                     for i in range(1, n)]
            # Add cofactor
            cofactor = ((-1) ** j) * mat[0][j] * determinant(minor)
            det += cofactor
        
        return det
    
    det = determinant(matrix)
    
    return det, trace