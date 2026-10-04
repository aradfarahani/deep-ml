import numpy as np

def xgboost_objective(gradients: np.ndarray, hessians: np.ndarray,
                      left_indices: np.ndarray, right_indices: np.ndarray,
                      lambda_reg: float = 1.0, gamma: float = 0.0) -> dict:
    """
    Calculate XGBoost objective function components for a potential split.
    
    Args:
        gradients: First-order gradients for each sample
        hessians: Second-order hessians for each sample
        left_indices: Indices of samples going to left child
        right_indices: Indices of samples going to right child
        lambda_reg: L2 regularization parameter
        gamma: Tree complexity penalty
        
    Returns:
        Dictionary with 'left_weight', 'right_weight', and 'gain'
    """
    # Calculate sum of gradients and hessians for left child
    G_L = np.sum(gradients[left_indices])
    H_L = np.sum(hessians[left_indices])
    
    # Calculate sum of gradients and hessians for right child
    G_R = np.sum(gradients[right_indices])
    H_R = np.sum(hessians[right_indices])
    
    # Calculate optimal leaf weights
    left_weight = -G_L / (H_L + lambda_reg)
    right_weight = -G_R / (H_R + lambda_reg)
    
    # Calculate gain from split
    score_left = (G_L ** 2) / (H_L + lambda_reg)
    score_right = (G_R ** 2) / (H_R + lambda_reg)
    score_parent = ((G_L + G_R) ** 2) / (H_L + H_R + lambda_reg)
    
    gain = 0.5 * (score_left + score_right - score_parent) - gamma
    
    return {
        'left_weight': float(round(left_weight, 4)),
        'right_weight': float(round(right_weight, 4)),
        'gain': float(round(gain, 4))
    }