import numpy as np

def multi_head_latent_attention(
    X: np.ndarray,
    W_dkv: np.ndarray,
    W_uk: np.ndarray,
    W_uv: np.ndarray,
    W_dq: np.ndarray,
    W_uq: np.ndarray,
    W_o: np.ndarray,
    n_heads: int
) -> tuple:
    """
    Perform Multi-Head Latent Attention (MLA).
    """
    seq_len, d_model = X.shape
    d_head = d_model // n_heads
    
    # Step 1: Compress KV into low-rank latent
    c_kv = X @ W_dkv  # (seq_len, d_c_kv)
    
    # Step 2: Reconstruct K and V from compressed latent
    K = c_kv @ W_uk  # (seq_len, d_model)
    V = c_kv @ W_uv  # (seq_len, d_model)
    
    # Step 3: Compress and reconstruct Q
    c_q = X @ W_dq   # (seq_len, d_c_q)
    Q = c_q @ W_uq   # (seq_len, d_model)
    
    # Step 4: Reshape for multi-head attention
    # (seq_len, d_model) -> (seq_len, n_heads, d_head) -> (n_heads, seq_len, d_head)
    Q = Q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    
    # Step 5: Scaled dot-product attention
    scale = np.sqrt(d_head)
    scores = Q @ K.transpose(0, 2, 1) / scale  # (n_heads, seq_len, seq_len)
    
    # Numerically stable softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Weighted sum of values
    attn_output = attn_weights @ V  # (n_heads, seq_len, d_head)
    
    # Step 6: Concatenate heads
    # (n_heads, seq_len, d_head) -> (seq_len, n_heads, d_head) -> (seq_len, d_model)
    attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, d_model)
    
    # Step 7: Output projection
    output = attn_output @ W_o  # (seq_len, d_model)
    
    return output, c_kv