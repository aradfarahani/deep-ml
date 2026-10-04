import numpy as np

def calculate_aic_bic(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> tuple:
    """
    Calculate AIC and BIC for model selection.
    
    Args:
        y_true: True target values
        y_pred: Predicted values from the model
        k: Number of parameters in the model
    
    Returns:
        Tuple of (AIC, BIC)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    
    # Calculate Residual Sum of Squares
    rss = np.sum((y_true - y_pred) ** 2)
    
    # Calculate AIC and BIC using simplified formulas for Gaussian models
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    
    return float(round(aic, 4)), float(round(bic, 4))