import numpy as np

def entropy_split_selection(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Find the best feature and threshold for splitting based on information gain.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
    
    Returns:
        Tuple of (best_feature_index, best_threshold, best_info_gain)
    """
    def calculate_entropy(labels):
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        # Only consider non-zero probabilities to avoid log(0)
        entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
        return entropy
    
    n_samples, n_features = X.shape
    parent_entropy = calculate_entropy(y)
    
    best_feature = 0
    best_threshold = 0.0
    best_info_gain = -1.0
    
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)
        
        if len(unique_values) < 2:
            continue
        
        # Consider midpoints between consecutive unique values as thresholds
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        
        for threshold in thresholds:
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            left_labels = y[left_mask]
            right_labels = y[right_mask]
            
            if len(left_labels) == 0 or len(right_labels) == 0:
                continue
            
            left_entropy = calculate_entropy(left_labels)
            right_entropy = calculate_entropy(right_labels)
            
            n_left = len(left_labels)
            n_right = len(right_labels)
            weighted_entropy = (n_left / n_samples) * left_entropy + \
                               (n_right / n_samples) * right_entropy
            
            info_gain = parent_entropy - weighted_entropy
            
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature_idx
                best_threshold = threshold
    
    return (best_feature, round(best_threshold, 4), round(best_info_gain, 4))