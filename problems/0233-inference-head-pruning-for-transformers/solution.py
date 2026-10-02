import numpy as np

def prune_attention_heads(
    attention_weights: np.ndarray,
    head_importance_scores: np.ndarray,
    pruning_ratio: float
) -> tuple[np.ndarray, list[int]]:
    """
    Prune less important attention heads.
    
    Args:
        attention_weights: Shape (num_heads, seq_len, seq_len)
        head_importance_scores: Shape (num_heads,)
        pruning_ratio: Fraction to prune (0.0 to 1.0)
    
    Returns:
        (pruned_attention_weights, kept_head_indices)
    """
    num_heads = attention_weights.shape[0]
    
    # Calculate number of heads to keep
    num_to_prune = int(num_heads * pruning_ratio)
    num_to_keep = num_heads - num_to_prune
    
    # Sort heads by importance (descending)
    head_indices = np.argsort(head_importance_scores)[::-1]
    kept_indices = head_indices[:num_to_keep]
    
    # Sort kept indices for consistent ordering
    kept_indices = np.sort(kept_indices)
    
    # Keep only important heads
    pruned_weights = attention_weights[kept_indices]
    
    return pruned_weights, kept_indices.tolist()