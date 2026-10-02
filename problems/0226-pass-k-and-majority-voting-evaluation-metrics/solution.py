import numpy as np
from collections import Counter

def pass_at_1(responses_correct: np.ndarray) -> float:
    """
    Compute pass@1 by averaging correctness across samples.
    
    Args:
        responses_correct: Boolean array indicating if each response is correct
        
    Returns:
        Average accuracy (pass@1)
    """
    return np.mean(responses_correct)


def majority_voting(responses: list[str]) -> str:
    """
    Return the most common response (consensus/majority vote).
    
    Args:
        responses: List of response strings
        
    Returns:
        Most frequent response
    """
    counts = Counter(responses)
    return counts.most_common(1)[0][0]


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute unbiased pass@k estimator.
    
    Formula: pass@k = 1 - C(n-c, k) / C(n, k)
    
    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k in pass@k
        
    Returns:
        Estimated pass@k
    """
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result