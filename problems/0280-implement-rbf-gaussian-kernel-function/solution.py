import numpy as np

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the RBF (Gaussian) kernel matrix between X1 and X2.
    
    Args:
        X1: First set of samples with shape (n1, d)
        X2: Second set of samples with shape (n2, d)
        gamma: Kernel coefficient (controls kernel width)
    
    Returns:
        Kernel matrix of shape (n1, n2)
    """
    # Compute squared norms of each sample
    X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)  # (n1, 1)
    X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)  # (n2, 1)
    
    # Compute squared Euclidean distances using expansion:
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
    sq_dist = X1_sq + X2_sq.T - 2 * np.dot(X1, X2.T)  # (n1, n2)
    
    # Clamp negative values to zero (numerical stability)
    sq_dist = np.maximum(sq_dist, 0)
    
    # Apply RBF kernel formula: K(x,y) = exp(-gamma * ||x-y||^2)
    K = np.exp(-gamma * sq_dist)
    
    return K