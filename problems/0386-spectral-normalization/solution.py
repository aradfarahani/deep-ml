import numpy as np

def spectral_normalization(W, num_iters=10, u_init=None):
    """
    Apply spectral normalization to a weight matrix using power iteration.
    
    Args:
        W: Weight matrix of shape (m, n)
        num_iters: Number of power iteration steps
        u_init: Optional initial vector of shape (m,)
    
    Returns:
        Tuple of (W_sn, sigma):
            W_sn: Spectrally normalized weight matrix
            sigma: Estimated spectral norm (largest singular value)
    """
    W = np.array(W, dtype=np.float64)
    m, n = W.shape
    
    if u_init is not None:
        u = np.array(u_init, dtype=np.float64)
    else:
        np.random.seed(0)
        u = np.random.randn(m)
    
    u = u / np.linalg.norm(u)
    
    for _ in range(num_iters):
        v = W.T @ u
        v = v / np.linalg.norm(v)
        u = W @ v
        u = u / np.linalg.norm(u)
    
    sigma = float(u @ W @ v)
    W_sn = W / sigma
    
    return W_sn, sigma