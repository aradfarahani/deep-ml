import numpy as np

def fit_gmm_1d(X, K, initial_means, initial_variances, initial_weights, n_iterations):
    """
    Fit a 1D Gaussian Mixture Model using the EM algorithm.
    
    Args:
        X: List of data points
        K: Number of mixture components
        initial_means: List of initial means for each component
        initial_variances: List of initial variances for each component
        initial_weights: List of initial mixture weights (should sum to 1)
        n_iterations: Number of EM iterations to run
    
    Returns:
        Dictionary with 'means', 'variances', 'weights' as lists rounded to 4 decimals
    """
    X = np.array(X, dtype=float)
    n = len(X)
    
    means = np.array(initial_means, dtype=float)
    variances = np.array(initial_variances, dtype=float)
    weights = np.array(initial_weights, dtype=float)
    
    MIN_VARIANCE = 1e-10  # Prevent division by zero
    
    for _ in range(n_iterations):
        # E-step: compute responsibilities
        responsibilities = np.zeros((n, K))
        for k in range(K):
            var_k = max(variances[k], MIN_VARIANCE)
            coef = 1.0 / np.sqrt(2 * np.pi * var_k)
            exponent = -0.5 * ((X - means[k]) ** 2) / var_k
            responsibilities[:, k] = weights[k] * coef * np.exp(exponent)
        
        # Normalize responsibilities
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        responsibilities = responsibilities / row_sums
        
        # M-step: update parameters
        Nk = responsibilities.sum(axis=0)
        
        for k in range(K):
            # Update mean
            means[k] = np.dot(responsibilities[:, k], X) / Nk[k]
            # Update variance
            variances[k] = np.dot(responsibilities[:, k], (X - means[k]) ** 2) / Nk[k]
            # Update weight
            weights[k] = Nk[k] / n
    
    return {
        'means': [round(float(m), 4) for m in means],
        'variances': [round(float(v), 4) for v in variances],
        'weights': [round(float(w), 4) for w in weights]
    }