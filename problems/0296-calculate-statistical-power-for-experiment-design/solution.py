import math

def calculate_power(effect_size: float, sample_size_per_group: int, alpha: float = 0.05, two_tailed: bool = True) -> float:
    """
    Calculate statistical power for a two-sample z-test.
    """
    def norm_cdf(x):
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def norm_ppf(p):
        """Inverse of standard normal CDF (quantile function)."""
        if p <= 0:
            return float('-inf')
        if p >= 1:
            return float('inf')
        if p < 0.5:
            return -norm_ppf(1 - p)
        
        # Rational approximation (Abramowitz and Stegun)
        t = math.sqrt(-2 * math.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)
    
    # Non-centrality parameter for two-sample test with equal group sizes
    ncp = effect_size * math.sqrt(sample_size_per_group / 2)
    
    if two_tailed:
        z_crit = norm_ppf(1 - alpha / 2)
        power = 1 - norm_cdf(z_crit - ncp) + norm_cdf(-z_crit - ncp)
    else:
        z_crit = norm_ppf(1 - alpha)
        power = 1 - norm_cdf(z_crit - ncp)
    
    return round(power, 4)