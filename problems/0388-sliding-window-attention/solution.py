import numpy as np

def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute sliding window attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Number of positions to the left and right each query can attend to
    
    Returns:
        Output matrix of shape (seq_len, d_v), rounded to 4 decimal places.
    """
    seq_len, d_k = Q.shape
    
    # Compute scaled attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Create sliding window mask
    mask = np.full((seq_len, seq_len), -np.inf)
    for i in range(seq_len):
        left = max(0, i - window_size)
        right = min(seq_len, i + window_size + 1)
        mask[i, left:right] = 0.0
    
    # Apply mask
    scores = scores + mask
    
    # Numerically stable softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Compute output
    output = attention_weights @ V
    
    return np.round(output, 4)