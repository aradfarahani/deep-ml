import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Get unique thresholds sorted in descending order
    unique_scores = np.unique(y_scores)
    thresholds = np.concatenate([[np.inf], np.sort(unique_scores)[::-1]])
    
    P = np.sum(y_true == 1)  # Total positives
    N = np.sum(y_true == 0)  # Total negatives
    
    fpr_list = []
    tpr_list = []
    
    for thresh in thresholds:
        # Classify samples with score >= threshold as positive
        y_pred = (y_scores >= thresh).astype(int)
        
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        
        tpr = TP / P if P > 0 else 0.0
        fpr = FP / N if N > 0 else 0.0
        
        tpr_list.append(round(float(tpr), 4))
        fpr_list.append(round(float(fpr), 4))
    
    return fpr_list, tpr_list