import numpy as np

def kl_divergence_estimator(pi_theta: np.ndarray, pi_ref: np.ndarray) -> np.ndarray:
    """
    Compute the unbiased KL divergence estimator from GRPO.
    
    Args:
        pi_theta: Current policy probabilities
        pi_ref: Reference policy probabilities
        
    Returns:
        Per-sample KL divergence estimates
    """
    ratio = pi_ref / pi_theta
    return ratio - np.log(ratio) - 1