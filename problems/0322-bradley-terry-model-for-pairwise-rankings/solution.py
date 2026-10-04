import numpy as np
from typing import List, Tuple

def fit_bradley_terry(comparisons: List[Tuple[int, int]], n_items: int, 
                      learning_rate: float = 0.5, n_iterations: int = 100) -> np.ndarray:
    """
    Fit Bradley-Terry model parameters using maximum likelihood estimation.
    
    Args:
        comparisons: List of (winner_idx, loser_idx) tuples
        n_items: Total number of items to rank
        learning_rate: Step size for gradient ascent
        n_iterations: Number of optimization iterations
    
    Returns:
        np.ndarray: Estimated strength parameters of shape (n_items,)
    """
    def sigmoid(x):
        # Numerically stable sigmoid
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    # Initialize parameters
    beta = np.zeros(n_items)
    
    # Gradient ascent on log-likelihood
    for _ in range(n_iterations):
        grad = np.zeros(n_items)
        
        for winner, loser in comparisons:
            # Probability that winner beats loser under current parameters
            prob = sigmoid(beta[winner] - beta[loser])
            
            # Gradient of log P(winner > loser) with respect to beta
            # d/d(beta_winner) log(sigmoid(x)) = 1 - sigmoid(x)
            # d/d(beta_loser) log(sigmoid(x)) = -(1 - sigmoid(x))
            grad[winner] += 1 - prob
            grad[loser] -= 1 - prob
        
        # Update parameters
        beta += learning_rate * grad
        
        # Center parameters for identifiability
        beta -= np.mean(beta)
    
    return beta