import numpy as np

def isolation_forest(X: np.ndarray, n_trees: int, sample_size: int, random_state: int = 42) -> np.ndarray:
    """
    Implement Isolation Forest for anomaly detection.
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    
    def c_factor(n):
        """Average path length of unsuccessful search in BST with n samples."""
        if n <= 1:
            return 0
        H = np.log(n - 1) + 0.5772156649
        return 2 * H - 2 * (n - 1) / n
    
    def build_tree(X_subset, depth, max_depth):
        n, m = X_subset.shape
        
        if depth >= max_depth or n <= 1:
            return {'type': 'leaf', 'size': n}
        
        feat = np.random.randint(m)
        min_v = X_subset[:, feat].min()
        max_v = X_subset[:, feat].max()
        
        if min_v == max_v:
            return {'type': 'leaf', 'size': n}
        
        split_val = np.random.uniform(min_v, max_v)
        
        left_idx = X_subset[:, feat] < split_val
        
        return {
            'type': 'node',
            'feature': feat,
            'split': split_val,
            'left': build_tree(X_subset[left_idx], depth + 1, max_depth),
            'right': build_tree(X_subset[~left_idx], depth + 1, max_depth)
        }
    
    def path_length(x, tree, depth):
        if tree['type'] == 'leaf':
            return depth + c_factor(tree['size'])
        
        if x[tree['feature']] < tree['split']:
            return path_length(x, tree['left'], depth + 1)
        return path_length(x, tree['right'], depth + 1)
    
    max_depth = int(np.ceil(np.log2(max(sample_size, 2))))
    trees = []
    
    for _ in range(n_trees):
        idx = np.random.choice(n_samples, size=min(sample_size, n_samples), replace=False)
        tree = build_tree(X[idx], 0, max_depth)
        trees.append(tree)
    
    avg_path_lengths = np.zeros(n_samples)
    for i in range(n_samples):
        total = 0
        for tree in trees:
            total += path_length(X[i], tree, 0)
        avg_path_lengths[i] = total / n_trees
    
    c = c_factor(sample_size)
    if c == 0:
        c = 1
    scores = 2 ** (-avg_path_lengths / c)
    
    return scores