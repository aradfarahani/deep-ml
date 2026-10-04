import numpy as np

def multiquery_attention(X: np.ndarray, W_queries: list, W_key: np.ndarray, W_value: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """
    Compute Multi-Query Attention.
    
    Args:
        X: Input array of shape (seq_len, d_model)
        W_queries: List of query weight matrices, each (d_model, d_k), one per head
        W_key: Shared key weight matrix of shape (d_model, d_k)
        W_value: Shared value weight matrix of shape (d_model, d_v)
        W_out: Output projection matrix of shape (num_heads * d_v, d_model)
    
    Returns:
        Output array of shape (seq_len, d_model), rounded to 4 decimal places
    """
    num_heads = len(W_queries)
    d_k = W_key.shape[1]
    
    # Shared key and value projections (computed once)
    K = X @ W_key    # (seq_len, d_k)
    V = X @ W_value  # (seq_len, d_v)
    
    head_outputs = []
    for h in range(num_heads):
        # Per-head query projection
        Q_h = X @ W_queries[h]  # (seq_len, d_k)
        
        # Scaled dot-product attention
        scores = Q_h @ K.T / np.sqrt(d_k)  # (seq_len, seq_len)
        
        # Numerically stable softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Weighted sum of values
        head_out = attn_weights @ V  # (seq_len, d_v)
        head_outputs.append(head_out)
    
    # Concatenate all heads
    concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, num_heads * d_v)
    
    # Output projection
    output = concat @ W_out  # (seq_len, d_model)
    
    return np.round(output, 4)