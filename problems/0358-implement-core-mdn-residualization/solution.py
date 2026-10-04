import numpy as np

def mdn_layer(f: np.ndarray, X: np.ndarray, sigma_inv: np.ndarray, N: int) -> np.ndarray:
    """
    Apply Metadata Normalization to features.
    """
    M = f.shape[0]
    E_xf = X.T @ f / M
    beta = N * sigma_inv @ E_xf
    r = f - X @ beta
    return r