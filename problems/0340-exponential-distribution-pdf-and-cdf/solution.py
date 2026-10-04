import numpy as np

def exponential_distribution(x: list, lam: float) -> dict:
    """
    Compute exponential distribution properties.
    
    Args:
        x: Points at which to evaluate PDF and CDF
        lam: Rate parameter (lambda) of the distribution
        
    Returns:
        Dictionary with 'pdf', 'cdf', 'mean', and 'variance' keys
    """
    # Handle invalid lambda
    if lam <= 0:
        return {'pdf': None, 'cdf': None, 'mean': None, 'variance': None}
    
    # Convert x to numpy array
    x = np.array(x, dtype=float)
    
    # Initialize arrays with zeros
    pdf = np.zeros_like(x, dtype=float)
    cdf = np.zeros_like(x, dtype=float)
    
    # For x >= 0: compute PDF and CDF
    valid_mask = x >= 0
    pdf[valid_mask] = lam * np.exp(-lam * x[valid_mask])
    cdf[valid_mask] = 1 - np.exp(-lam * x[valid_mask])
    
    # Compute mean and variance
    mean = 1.0 / lam
    variance = 1.0 / (lam ** 2)
    
    return {
        'pdf': np.round(pdf, 4).tolist(),
        'cdf': np.round(cdf, 4).tolist(),
        'mean': round(mean, 4),
        'variance': round(variance, 4)
    }