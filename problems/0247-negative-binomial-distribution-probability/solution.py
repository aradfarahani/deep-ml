import math

def negative_binomial_pmf(k: int, r: int, p: float) -> float:
    """
    Calculate the probability of observing exactly k failures
    before achieving r successes in independent Bernoulli trials.
    
    Args:
        k: Number of failures (non-negative integer)
        r: Number of successes required (positive integer)
        p: Probability of success on each trial (0 < p <= 1)
    
    Returns:
        Probability P(X = k) rounded to 5 decimal places
    """
    # Calculate binomial coefficient C(k + r - 1, k)
    binom_coeff = math.comb(k + r - 1, k)
    
    # Calculate probability: C(k+r-1, k) * p^r * (1-p)^k
    probability = binom_coeff * (p ** r) * ((1 - p) ** k)
    
    return round(probability, 5)