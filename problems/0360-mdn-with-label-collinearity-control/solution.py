import numpy as np

def mdn_with_collinearity(f: np.ndarray, X: np.ndarray, y: np.ndarray, 
                          sigma_tilde_inv: np.ndarray, N: int) -> np.ndarray:
    """
    Apply MDN with label collinearity control.
    """
    M = f.shape[0]
    K = X.shape[1]
    
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    X_tilde = np.hstack([X, y])
    E_tilde = X_tilde.T @ f / M
    beta_tilde = N * sigma_tilde_inv @ E_tilde
    beta_X = beta_tilde[:K, :]
    r = f - X @ beta_X
    
    return r