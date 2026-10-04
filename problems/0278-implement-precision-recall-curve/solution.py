import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Get unique thresholds sorted in descending order
    thresholds = np.sort(np.unique(y_scores))[::-1]
    
    precisions = []
    recalls = []
    
    total_positives = np.sum(y_true)
    
    for thresh in thresholds:
        # Samples with score >= threshold are predicted positive
        y_pred = (y_scores >= thresh).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / total_positives if total_positives > 0 else 0.0
        
        precisions.append(float(precision))
        recalls.append(float(recall))
    
    return precisions, recalls, thresholds.tolist()