import numpy as np

def calculate_oob_score(n_samples: int, bootstrap_indices: list, predictions: list, y_true: list) -> float:
    """
    Calculate the Out-of-Bag score for a bagging ensemble.
    
    Args:
        n_samples: Total number of samples in the dataset
        bootstrap_indices: List of lists containing indices used to train each estimator
        predictions: List of lists containing predictions from each estimator for all samples
        y_true: True labels for all samples
    
    Returns:
        OOB accuracy score as a float
    """
    n_estimators = len(bootstrap_indices)
    oob_predictions = [[] for _ in range(n_samples)]
    
    # Collect OOB predictions for each sample
    for est_idx in range(n_estimators):
        in_bag = set(bootstrap_indices[est_idx])
        for sample_idx in range(n_samples):
            if sample_idx not in in_bag:
                oob_predictions[sample_idx].append(predictions[est_idx][sample_idx])
    
    # Compute accuracy using majority voting
    correct = 0
    n_oob_samples = 0
    
    for sample_idx in range(n_samples):
        if len(oob_predictions[sample_idx]) > 0:
            votes = np.array(oob_predictions[sample_idx])
            unique, counts = np.unique(votes, return_counts=True)
            final_pred = unique[np.argmax(counts)]
            
            if final_pred == y_true[sample_idx]:
                correct += 1
            n_oob_samples += 1
    
    if n_oob_samples == 0:
        return 0.0
    
    return round(correct / n_oob_samples, 4)