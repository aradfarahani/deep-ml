import math

def hypergeometric_pmf(N: int, K: int, n: int, k: int) -> float:
    """
    Calculate the PMF of the hypergeometric distribution.
    
    Args:
        N: Total population size
        K: Number of success states in population
        n: Number of draws (without replacement)
        k: Number of observed successes
    
    Returns:
        float: P(X = k), rounded to 4 decimal places
    """
    # Check if k is in valid range
    # k must be at least max(0, n - (N - K)) and at most min(K, n)
    k_min = max(0, n - (N - K))
    k_max = min(K, n)
    
    if k < k_min or k > k_max:
        return 0.0
    
    # Calculate using binomial coefficients
    # P(X = k) = C(K, k) * C(N-K, n-k) / C(N, n)
    numerator = math.comb(K, k) * math.comb(N - K, n - k)
    denominator = math.comb(N, n)
    
    return round(numerator / denominator, 4)