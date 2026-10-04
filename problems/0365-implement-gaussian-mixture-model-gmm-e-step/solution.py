import numpy as np

def gmm_e_step(X: np.ndarray, means: np.ndarray, variances: np.ndarray, 
               mixing_coeffs: np.ndarray) -> np.ndarray:
    """
    Compute the E-step of Gaussian Mixture Model.
    
    Args:
        X: Data points of shape (n_samples,)
        means: Component means of shape (n_components,)
        variances: Component variances of shape (n_components,)
        mixing_coeffs: Mixing coefficients of shape (n_components,)
    
    Returns:
        Responsibility matrix of shape (n_samples, n_components)
    """
    n_samples = len(X)
    n_components = len(means)
    
    # Initialize matrix to store weighted likelihoods
    weighted_likelihoods = np.zeros((n_samples, n_components))
    
    # Compute weighted likelihood for each component
    for k in range(n_components):
        # Gaussian PDF coefficient
        coeff = 1.0 / np.sqrt(2 * np.pi * variances[k])
        # Exponent term
        exponent = -0.5 * ((X - means[k]) ** 2) / variances[k]
        # Weighted likelihood = mixing_coeff * N(x | mu, sigma^2)
        weighted_likelihoods[:, k] = mixing_coeffs[k] * coeff * np.exp(exponent)
    
    # Normalize to get responsibilities (each row sums to 1)
    responsibilities = weighted_likelihoods / weighted_likelihoods.sum(axis=1, keepdims=True)
    
    return responsibilities