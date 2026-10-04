def random_forest_feature_importance(trees: list, n_features: int) -> list:
    """
    Calculate feature importance from a random forest using Mean Decrease in Impurity.
    
    Args:
        trees: List of trees, where each tree is a list of node splits.
               Each split is a dict with:
               - 'feature_index': int, the feature used for splitting
               - 'impurity_decrease': float, the weighted impurity decrease
        n_features: Total number of features in the dataset
    
    Returns:
        List of feature importances normalized to sum to 1.0
    """
    import numpy as np
    
    feature_importance = np.zeros(n_features)
    
    for tree in trees:
        for split in tree:
            feature_idx = split['feature_index']
            impurity_decrease = split['impurity_decrease']
            feature_importance[feature_idx] += impurity_decrease
    
    # Normalize to sum to 1
    total = feature_importance.sum()
    if total > 0:
        feature_importance = feature_importance / total
    
    return feature_importance.tolist()