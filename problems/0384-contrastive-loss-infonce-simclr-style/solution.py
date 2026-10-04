import numpy as np

def contrastive_loss(embeddings: np.ndarray, temperature: float) -> float:
    """
    Compute the NT-Xent (SimCLR-style) contrastive loss.
    
    Args:
        embeddings: Array of shape (2N, d) where consecutive pairs
                    (2i, 2i+1) are positive pairs.
        temperature: Temperature scaling parameter (tau > 0).
    
    Returns:
        The mean contrastive loss as a float.
    """
    n = embeddings.shape[0]  # 2N
    
    # L2 normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    
    # Cosine similarity matrix scaled by temperature
    sim = np.dot(normalized, normalized.T) / temperature
    
    # Build positive pair index mapping: (0,1), (2,3), (4,5), ...
    pos_idx = np.empty(n, dtype=int)
    pos_idx[0::2] = np.arange(1, n, 2)
    pos_idx[1::2] = np.arange(0, n, 2)
    
    loss = 0.0
    for i in range(n):
        j = pos_idx[i]
        pos_sim = sim[i, j]
        
        # Gather logits for all k != i
        logits = np.concatenate([sim[i, :i], sim[i, i+1:]])
        
        # Log-sum-exp trick for numerical stability
        max_logit = np.max(logits)
        log_sum_exp = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
        
        loss += -pos_sim + log_sum_exp
    
    return loss / n