import numpy as np

def jensen_shannon_divergence(P: list[float], Q: list[float]) -> float:
    """
    Compute the Jensen-Shannon Divergence between two probability distributions.
    
    Args:
        P: First probability distribution (must sum to 1)
        Q: Second probability distribution (must sum to 1)
    
    Returns:
        Jensen-Shannon Divergence value
    """
    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)
    
    # Compute the average distribution M
    M = 0.5 * (P + Q)
    
    # Compute KL divergence for each half
    def kl_divergence(p, q):
        # Add small epsilon for numerical stability
        epsilon = 1e-10
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)
        return np.sum(p * np.log(p / q))
    
    # Jensen-Shannon Divergence
    jsd = 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)
    
    return float(jsd)