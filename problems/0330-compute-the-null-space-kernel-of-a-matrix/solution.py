import numpy as np

def compute_null_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute an orthonormal basis for the null space (kernel) of matrix A.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering singular values as zero
    
    Returns:
        Matrix of shape (n, k) where k is the dimension of the null space.
        Columns form an orthonormal basis for the null space.
    """
    m, n = A.shape
    
    # Compute the full SVD
    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    
    # V is the conjugate transpose of Vh
    V = Vh.T
    
    # Determine numerical rank by counting significant singular values
    rank = np.sum(s > tol)
    
    # The null space is spanned by the last (n - rank) columns of V
    # These correspond to zero (or near-zero) singular values
    null_space_basis = V[:, rank:]
    
    return null_space_basis