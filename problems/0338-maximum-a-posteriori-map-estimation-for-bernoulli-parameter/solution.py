import numpy as np

def map_estimate_bernoulli(observations: list, alpha: float, beta: float) -> float:
    """
    Compute the Maximum A Posteriori (MAP) estimate for a Bernoulli parameter.
    
    Args:
        observations: List of binary observations (0s and 1s)
        alpha: Alpha parameter of Beta prior (>= 1)
        beta: Beta parameter of Beta prior (>= 1)
    
    Returns:
        MAP estimate of the probability parameter, rounded to 4 decimal places
    """
    observations = np.array(observations, dtype=float)
    n = len(observations)
    k = np.sum(observations)  # number of successes (1s)
    
    # Posterior is Beta(alpha + k, beta + n - k)
    alpha_post = alpha + k
    beta_post = beta + (n - k)
    
    # Mode of Beta(a, b) distribution:
    # - When a > 1 and b > 1: mode = (a - 1) / (a + b - 2)
    # - When a = 1 and b = 1: uniform, use 0.5 as convention
    # - When a <= 1 and b > 1: mode = 0
    # - When a > 1 and b <= 1: mode = 1
    
    if alpha_post > 1 and beta_post > 1:
        map_est = (alpha_post - 1) / (alpha_post + beta_post - 2)
    elif alpha_post == 1 and beta_post == 1:
        map_est = 0.5  # Uniform posterior
    elif alpha_post <= 1:
        map_est = 0.0  # Mode at lower boundary
    else:  # beta_post <= 1
        map_est = 1.0  # Mode at upper boundary
    
    return round(map_est, 4)