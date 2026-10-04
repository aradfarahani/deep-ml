import math

def beta_distribution_stats(x: float, alpha: float, beta_param: float) -> dict:
    """
    Compute Beta distribution statistics.
    
    Args:
        x: Value at which to evaluate the PDF
        alpha: First shape parameter (alpha > 0)
        beta_param: Second shape parameter (beta > 0)
    
    Returns:
        Dictionary with 'pdf', 'mean', and 'variance'
    """
    # Compute the Beta function using the gamma function
    # B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b)
    beta_func = math.gamma(alpha) * math.gamma(beta_param) / math.gamma(alpha + beta_param)
    
    # Compute PDF: f(x) = x^(alpha-1) * (1-x)^(beta-1) / B(alpha, beta)
    if x <= 0 or x >= 1:
        pdf = 0.0
    else:
        pdf = (x ** (alpha - 1)) * ((1 - x) ** (beta_param - 1)) / beta_func
    
    # Compute mean: E[X] = alpha / (alpha + beta)
    mean = alpha / (alpha + beta_param)
    
    # Compute variance: Var[X] = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
    variance = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))
    
    return {
        'pdf': pdf,
        'mean': mean,
        'variance': variance
    }