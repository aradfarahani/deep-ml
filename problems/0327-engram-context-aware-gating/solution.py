import numpy as np

def engram_context_gating(h: np.ndarray, e: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Implement Engram context-aware gating mechanism.
    
    Args:
        h: Hidden states of shape (T, d)
        e: Retrieved memory embeddings of shape (T, d_mem)
        W_K: Key projection matrix of shape (d_mem, d)
        W_V: Value projection matrix of shape (d_mem, d)
        eps: Small constant for numerical stability in RMSNorm
    
    Returns:
        Gated output of shape (T, d)
    """
    # Get dimension for scaling
    d = h.shape[-1]
    
    # Step 1: Project memory to get key and value
    k = e @ W_K  # (T, d)
    v = e @ W_V  # (T, d)
    
    # Step 2: Apply RMSNorm to query (h) and key (k)
    h_rms = np.sqrt(np.mean(h ** 2, axis=-1, keepdims=True) + eps)
    h_norm = h / h_rms
    
    k_rms = np.sqrt(np.mean(k ** 2, axis=-1, keepdims=True) + eps)
    k_norm = k / k_rms
    
    # Step 3: Compute gating scalar with scaled dot product
    # For each position t: alpha_t = sigmoid(h_norm[t] Â· k_norm[t] / sqrt(d))
    dot_product = np.sum(h_norm * k_norm, axis=-1, keepdims=True)  # (T, 1)
    scaled_dot = dot_product / np.sqrt(d)
    alpha = 1.0 / (1.0 + np.exp(-scaled_dot))  # sigmoid, shape (T, 1)
    
    # Step 4: Return gated value
    gated_output = alpha * v  # (T, d)
    
    return gated_output