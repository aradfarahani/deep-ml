import numpy as np

def gated_attention(
    X: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_g: np.ndarray
) -> np.ndarray:
    """
    Compute Gated Attention output.
    """
    # Compute Q, K, V projections
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    
    # Get dimension for scaling
    d_k = Q.shape[1]
    
    # Compute attention scores and weights
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Softmax (numerically stable)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Standard attention output
    Y = attention_weights @ V
    
    # Compute gate (sigmoid)
    gate_input = X @ W_g
    G = 1 / (1 + np.exp(-gate_input))
    
    # Apply gate element-wise
    Y_gated = G * Y
    
    return np.round(Y_gated, 4)