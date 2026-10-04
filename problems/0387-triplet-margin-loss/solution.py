import numpy as np

def triplet_margin_loss(anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 1.0) -> float:
    """
    Compute the triplet margin loss for metric learning.
    
    Args:
        anchor: Anchor embeddings, shape (D,) for single or (N, D) for batch
        positive: Positive embeddings (same class as anchor), same shape as anchor
        negative: Negative embeddings (different class from anchor), same shape as anchor
        margin: Minimum desired distance gap between positive and negative pairs
    
    Returns:
        Mean triplet margin loss as a float
    """
    anchor = np.array(anchor, dtype=np.float64)
    positive = np.array(positive, dtype=np.float64)
    negative = np.array(negative, dtype=np.float64)
    
    # Handle single triplet (1D) by adding batch dimension
    if anchor.ndim == 1:
        anchor = anchor[np.newaxis, :]
        positive = positive[np.newaxis, :]
        negative = negative[np.newaxis, :]
    
    # Compute Euclidean distances
    d_pos = np.sqrt(np.sum((anchor - positive) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((anchor - negative) ** 2, axis=1))
    
    # Compute per-triplet loss with margin and clamp at zero
    losses = np.maximum(0.0, d_pos - d_neg + margin)
    
    # Return mean loss
    return float(np.mean(losses))