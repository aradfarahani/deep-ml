import numpy as np

def matthews_correlation_coefficient(y_true: list, y_pred: list) -> float:
    """
    Calculate the Matthews Correlation Coefficient for binary classification.
    
    Args:
        y_true: List of actual binary labels (0 or 1)
        y_pred: List of predicted binary labels (0 or 1)
    
    Returns:
        MCC value rounded to 4 decimal places
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix components
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate MCC
    numerator = TP * TN - FP * FN
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    if denominator == 0:
        return 0.0
    
    mcc = numerator / denominator
    return round(float(mcc), 4)