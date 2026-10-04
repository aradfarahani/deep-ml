import numpy as np

def distance_correlation_squared(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute squared distance correlation between X and Y.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    def pairwise_distances(Z):
        sq_norms = np.sum(Z**2, axis=1, keepdims=True)
        return np.sqrt(np.maximum(sq_norms + sq_norms.T - 2 * Z @ Z.T, 0))
    
    def double_center(D):
        row_mean = D.mean(axis=1, keepdims=True)
        col_mean = D.mean(axis=0, keepdims=True)
        grand_mean = D.mean()
        return D - row_mean - col_mean + grand_mean
    
    A = double_center(pairwise_distances(X))
    B = double_center(pairwise_distances(Y))
    
    dCov_sq = (A * B).mean()
    dVar_X_sq = (A * A).mean()
    dVar_Y_sq = (B * B).mean()
    
    if dVar_X_sq * dVar_Y_sq == 0:
        return 0.0
    
    return float(dCov_sq / np.sqrt(dVar_X_sq * dVar_Y_sq))