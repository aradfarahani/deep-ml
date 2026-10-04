import numpy as np

def mhc_forward(
    x: np.ndarray,
    H_pre_raw: np.ndarray,
    H_post_raw: np.ndarray,
    H_res_raw: np.ndarray,
    layer_output: np.ndarray,
    sinkhorn_iters: int = 5
) -> np.ndarray:
    """
    Compute mHC forward pass with manifold-constrained mappings.
    """
    # Apply sigmoid constraint to H_post (ensures non-negative)
    H_post = 2 * (1 / (1 + np.exp(-H_post_raw)))  # Shape: (1, n)
    
    # Apply Sinkhorn iteration to get doubly stochastic H_res
    H_res = np.exp(H_res_raw)  # Make positive, shape: (n, n)
    for _ in range(sinkhorn_iters):
        # Row normalization
        H_res = H_res / H_res.sum(axis=1, keepdims=True)
        # Column normalization
        H_res = H_res / H_res.sum(axis=0, keepdims=True)
    
    # mHC forward: x_out = H_res @ x + H_post.T @ layer_output
    # H_res @ x: mix the n streams, shape (n, C)
    # H_post.T @ layer_output: broadcast layer output to streams, shape (n, C)
    x_out = H_res @ x + H_post.T @ layer_output
    
    return x_out