import numpy as np

def decision_tree_regressor(X_train, y_train, X_test, max_depth=2, min_samples_split=2):
    """
    Build a decision tree for regression and predict on test data.
    
    Args:
        X_train: Training features, shape (n_samples, n_features)
        y_train: Training targets, shape (n_samples,)
        X_test: Test features, shape (m_samples, n_features)
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
    
    Returns:
        List of predictions for X_test, rounded to 4 decimal places
    """
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    
    def calculate_mse(y):
        if len(y) == 0:
            return 0.0
        return np.mean((y - np.mean(y))**2)
    
    def find_best_split(X, y):
        n_samples, n_features = X.shape
        if n_samples < min_samples_split:
            return None
        
        current_mse = calculate_mse(y)
        best_gain = 0
        best_split = None
        
        for feature_idx in range(n_features):
            values = np.unique(X[:, feature_idx])
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_mse = calculate_mse(y[left_mask])
                right_mse = calculate_mse(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_mse = (n_left * left_mse + n_right * right_mse) / n_samples
                
                gain = current_mse - weighted_mse
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, threshold)
        
        return best_split
    
    def build_tree(X, y, depth):
        if depth >= max_depth or len(y) < min_samples_split or len(np.unique(y)) == 1:
            return {'leaf': True, 'value': float(np.mean(y))}
        
        split = find_best_split(X, y)
        
        if split is None:
            return {'leaf': True, 'value': float(np.mean(y))}
        
        feature_idx, threshold = split
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        return {
            'leaf': False,
            'feature': feature_idx,
            'threshold': threshold,
            'left': build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': build_tree(X[right_mask], y[right_mask], depth + 1)
        }
    
    def predict_single(node, x):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return predict_single(node['left'], x)
        else:
            return predict_single(node['right'], x)
    
    tree = build_tree(X_train, y_train, 0)
    predictions = [predict_single(tree, x) for x in X_test]
    
    return [round(p, 4) for p in predictions]