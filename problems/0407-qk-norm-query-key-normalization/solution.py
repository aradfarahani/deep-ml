import numpy as np

def qk_norm_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, temperature: float = 1.0) -> tuple:
    """
    Apply QK-Norm attention: L2-normalize queries and keys before computing
    scaled dot-product attention.
    
    Args:
        Q: Query matrix, shape (seq_len_q, d_k)
        K: Key matrix, shape (seq_len_k, d_k)
        V: Value matrix, shape (seq_len_k, d_v)
        temperature: Temperature scaling parameter (default: 1.0)
    
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    eps = 1e-8
    # L2 normalize each query vector (row-wise)
    Q_norms = np.linalg.norm(Q, axis=-1, keepdims=True)
    Q_normalized = Q / (Q_norms + eps)
    # L2 normalize each key vector (row-wise)
    K_norms = np.linalg.norm(K, axis=-1, keepdims=True)
    K_normalized = K / (K_norms + eps)
    # Compute attention scores with temperature scaling
    scores = Q_normalized @ K_normalized.T / temperature
    # Numerically stable softmax
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    # Compute attention output
    attention_output = attention_weights @ V
    return attention_output, attention_weights