import numpy as np

def svm_margin_width(w: np.ndarray) -> float:
    """
    Calculate the margin width of a linear SVM classifier.
    
    Parameters:
    w : np.ndarray - weight vector defining the hyperplane
    
    Returns:
    float - the total margin width
    """
    norm_w = np.linalg.norm(w)
    return 2.0 / norm_w