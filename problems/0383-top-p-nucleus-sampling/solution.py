import numpy as np

def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    logits = np.array(logits, dtype=float)
    
    # Convert logits to probabilities via softmax
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits)
    
    # Sort indices by probability in descending order (stable sort preserves index order for ties)
    sorted_indices = np.argsort(-probs, kind='stable')
    sorted_probs = probs[sorted_indices]
    
    # Compute cumulative probabilities
    cumulative_probs = np.cumsum(sorted_probs)
    
    # Find nucleus size: smallest set where cumulative prob >= p
    nucleus_size = len(logits)
    for i in range(len(cumulative_probs)):
        if cumulative_probs[i] >= p:
            nucleus_size = i + 1
            break
    
    # Build filtered distribution
    nucleus_indices = sorted_indices[:nucleus_size]
    filtered = np.zeros_like(probs)
    filtered[nucleus_indices] = probs[nucleus_indices]
    
    # Renormalize
    filtered = filtered / np.sum(filtered)
    
    return [round(float(x), 4) for x in filtered]