import numpy as np

def stacking_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                        base_classifiers: list, meta_classifier, n_folds: int = 5) -> np.ndarray:
    """
    Implement a stacking classifier ensemble.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,)
        X_test: Test features of shape (m_samples, n_features)
        base_classifiers: List of classifier functions
        meta_classifier: Meta-level classifier function
        n_folds: Number of cross-validation folds
    
    Returns:
        np.ndarray: Final predictions on X_test
    """
    n_samples = X_train.shape[0]
    n_base = len(base_classifiers)
    
    # Initialize arrays for meta-features
    meta_train = np.zeros((n_samples, n_base))
    meta_test = np.zeros((X_test.shape[0], n_base))
    
    # Calculate fold size
    fold_size = n_samples // n_folds
    
    # Generate meta-features using cross-validation
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
        
        val_idx = np.arange(val_start, val_end)
        train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n_samples)])
        
        X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
        X_fold_val = X_train[val_idx]
        
        for clf_idx, clf in enumerate(base_classifiers):
            preds = clf(X_fold_train, y_fold_train, X_fold_val)
            meta_train[val_idx, clf_idx] = preds
    
    # Train base classifiers on full training data and predict on test
    for clf_idx, clf in enumerate(base_classifiers):
        meta_test[:, clf_idx] = clf(X_train, y_train, X_test)
    
    # Train meta-classifier and make final predictions
    final_predictions = meta_classifier(meta_train, y_train, meta_test)
    
    return final_predictions