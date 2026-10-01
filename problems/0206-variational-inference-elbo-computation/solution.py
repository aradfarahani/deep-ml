import numpy as np

def compute_elbo(x: list[float], q_mean: float, q_std: float, 
                 prior_mean: float, prior_std: float,
                 likelihood_std: float, n_samples: int = 1000) -> float:
    """
    Compute the Evidence Lower Bound (ELBO) for variational inference.
    
    Args:
        x: Observed data points
        q_mean: Mean of variational distribution q(z)
        q_std: Standard deviation of variational distribution q(z)
        prior_mean: Mean of prior distribution p(z)
        prior_std: Standard deviation of prior distribution p(z)
        likelihood_std: Standard deviation of likelihood p(x|z)
        n_samples: Number of Monte Carlo samples
    
    Returns:
        ELBO value (float)
    """
    x = np.array(x)
    
    def gaussian_log_pdf(val, mean, std):
        """Compute log PDF of Gaussian distribution."""
        return -0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((val - mean) / std)**2
    
    # Sample from variational distribution q(z)
    z_samples = np.random.normal(q_mean, q_std, n_samples)
    
    # Compute E_q[log p(x|z)] - expected log likelihood
    expected_log_likelihood = np.mean([
        np.sum(gaussian_log_pdf(x, z, likelihood_std)) for z in z_samples
    ])
    
    # Compute E_q[log p(z)] - expected log prior
    expected_log_prior = np.mean([
        gaussian_log_pdf(z, prior_mean, prior_std) for z in z_samples
    ])
    
    # Compute H[q] - entropy of variational distribution (analytic for Gaussian)
    entropy_q = 0.5 * np.log(2 * np.pi * np.e * q_std**2)
    
    # ELBO = E_q[log p(x|z)] + E_q[log p(z)] + H[q]
    elbo = expected_log_likelihood + expected_log_prior + entropy_q
    
    return float(elbo)