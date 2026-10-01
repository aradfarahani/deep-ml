import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
    """
    Compute the mutual information between two random variables.
    
    Args:
        joint_prob: 2D joint probability distribution P(X,Y)
    
    Returns:
        Mutual information I(X;Y)
    """
    joint = np.array(joint_prob, dtype=float)
    
    # Compute marginal distributions
    p_x = np.sum(joint, axis=1)  # Sum over Y
    p_y = np.sum(joint, axis=0)  # Sum over X
    
    # Compute mutual information
    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (p_x[i] * p_y[j]))
    
    return float(mi)