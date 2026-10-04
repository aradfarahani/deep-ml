import numpy as np

def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    """
    Perform one step of gradient boosting regression using a decision stump.
    
    Args:
        X: Feature matrix (list of lists), shape (n_samples, n_features)
        y: Target values (list), shape (n_samples,)
        current_predictions: Current ensemble predictions (list), shape (n_samples,)
        learning_rate: Learning rate for the update (default 0.1)
    
    Returns:
        List of updated predictions rounded to 4 decimal places
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    current_predictions = np.array(current_predictions, dtype=float)
    
    # Calculate residuals (negative gradient of MSE loss)
    residuals = y - current_predictions
    
    n_samples = X.shape[0]
    n_features = X.shape[1] if X.ndim > 1 else 1
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    best_mse = float('inf')
    best_split = None
    
    # Find the best split across all features
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        unique_values = np.sort(np.unique(feature_values))
        
        # Try all possible thresholds (midpoints between consecutive values)
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            # Predict mean residual in each partition
            left_pred = np.mean(residuals[left_mask])
            right_pred = np.mean(residuals[right_mask])
            
            # Calculate MSE for this split
            predictions = np.where(left_mask, left_pred, right_pred)
            mse = np.mean((residuals - predictions) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_split = {
                    'feature': feature_idx,
                    'threshold': threshold,
                    'left_value': left_pred,
                    'right_value': right_pred
                }
    
    # Make predictions with the stump
    if best_split is None:
        # No valid split found, predict mean of all residuals
        stump_predictions = np.full(n_samples, np.mean(residuals))
    else:
        feature_values = X[:, best_split['feature']]
        stump_predictions = np.where(
            feature_values <= best_split['threshold'],
            best_split['left_value'],
            best_split['right_value']
        )
    
    # Update predictions: new = current + learning_rate * stump
    new_predictions = current_predictions + learning_rate * stump_predictions
    
    return [round(float(p), 4) for p in new_predictions]