import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (Ï_i) = Ï_theta(o_i | q) / Ï_theta_old(o_i | q).
        A: List of advantage estimates (A_i).
        pi_theta_old: List representing the old policy probabilities Ï_theta_old(o_i | q).
        pi_theta_ref: List representing the reference policy probabilities Ï_ref(o_i | q).
        epsilon: Clipping parameter (Ïµ).
        beta: KL divergence penalty coefficient (Î²).

    Returns:
        The computed GRPO objective value.
    """
    G = len(rhos)
    if not (len(A) == len(pi_theta_old) == len(pi_theta_ref) == G):
        raise ValueError("All input lists must have the same length.")
    
    # Compute clipped likelihood ratios
    clipped_rhos = np.clip(rhos, 1 - epsilon, 1 + epsilon)
    
    # Compute the minimum terms for the objective
    unclipped = np.array(rhos) 