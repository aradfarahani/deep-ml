import numpy as np

def covariance_from_joint_pmf(x_values: list, y_values: list, joint_pmf: np.ndarray) -> float:
    """
    Compute the covariance of X and Y from their joint PMF.
    
    Args:
        x_values: List of possible values for X
        y_values: List of possible values for Y
        joint_pmf: 2D numpy array where joint_pmf[i][j] = P(X=x_values[i], Y=y_values[j])
    
    Returns:
        Covariance of X and Y as a float
    """
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    
    # Compute marginal PMFs
    p_x = np.sum(joint_pmf, axis=1)  # Sum over y for each x
    p_y = np.sum(joint_pmf, axis=0)  # Sum over x for each y
    
    # Compute expected values E[X] and E[Y]
    E_X = np.sum(x_values * p_x)
    E_Y = np.sum(y_values * p_y)
    
    # Compute E[XY]
    E_XY = 0.0
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            E_XY += x * y * joint_pmf[i][j]
    
    # Covariance = E[XY] - E[X]*E[Y]
    cov = E_XY - E_X * E_Y
    return float(cov)