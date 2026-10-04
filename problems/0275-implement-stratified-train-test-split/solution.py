import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
    """
    Split data into train and test sets while maintaining class proportions.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    X = np.array(X)
    y = np.array(y)
    
    classes = np.unique(y)
    
    train_indices = []
    test_indices = []
    
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        n_cls = len(cls_indices)
        
        shuffled = cls_indices.copy()
        np.random.shuffle(shuffled)
        
        n_test = int(n_cls * test_size)
        
        test_indices.extend(shuffled[:n_test].tolist())
        train_indices.extend(shuffled[n_test:].tolist())
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test