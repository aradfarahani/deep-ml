import numpy as np

def irope_attention(Q: list, K: list, V: list, positions: list, layer_index: int, rope_layers: list, base: float = 10000.0) -> dict:
    Q = np.array(Q, dtype=np.float64)
    K = np.array(K, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    positions = np.array(positions, dtype=np.float64)
    
    seq_len, d_head = Q.shape
    uses_rope = layer_index in rope_layers
    
    if uses_rope:
        Q_out = _apply_rope(Q, positions, d_head, base)
        K_out = _apply_rope(K, positions, d_head, base)
    else:
        Q_out = Q.copy()
        K_out = K.copy()
    
    scores = Q_out @ K_out.T / np.sqrt(d_head)
    
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    output = attention_weights @ V
    
    return {
        'output': np.round(output, 4).tolist(),
        'attention_weights': np.round(attention_weights, 4).tolist(),
        'uses_rope': uses_rope
    }

def _apply_rope(X, positions, d_head, base):
    seq_len = X.shape[0]
    X_rot = np.zeros_like(X)
    
    for i in range(d_head // 2):
        theta = 1.0 / (base ** (2.0 * i / d_head))
        angles = positions * theta
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        
        X_rot[:, 2*i] = X[:, 2*i] * cos_a - X[:, 2*i+1] * sin_a
        X_rot[:, 2*i+1] = X[:, 2*i] * sin_a + X[:, 2*i+1] * cos_a
    
    return X_rot