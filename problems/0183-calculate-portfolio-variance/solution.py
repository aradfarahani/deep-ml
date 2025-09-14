import numpy as np

def calculate_portfolio_variance(cov_matrix: list[list[float]], weights: list[float]) -> float:
    cov_matrix = np.array(cov_matrix)
    weights = np.array(weights)
    variance = float(weights.T @ cov_matrix @ weights)
    return variance