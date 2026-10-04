import numpy as np

def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Calinski-Harabasz Index for clustering evaluation.
    
    Args:
        X: numpy array of shape (n_samples, n_features) containing data points
        labels: numpy array of shape (n_samples,) containing cluster assignments
    
    Returns:
        float: Calinski-Harabasz score (higher is better)
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    # Edge cases: single cluster or each point is its own cluster
    if k == 1 or k == n_samples:
        return 0.0
    
    # Compute global centroid
    global_centroid = np.mean(X, axis=0)
    
    # Compute within-cluster and between-cluster dispersion
    W = 0.0  # Within-cluster dispersion
    B = 0.0  # Between-cluster dispersion
    
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_points = X[cluster_mask]
        cluster_centroid = np.mean(cluster_points, axis=0)
        n_cluster = len(cluster_points)
        
        # Within-cluster: sum of squared distances to cluster centroid
        W += np.sum((cluster_points - cluster_centroid) ** 2)
        
        # Between-cluster: weighted squared distance from cluster centroid to global centroid
        B += n_cluster * np.sum((cluster_centroid - global_centroid) ** 2)
    
    # Calinski-Harabasz Index formula
    ch_score = (B / (k - 1)) / (W / (n_samples - k))
    
    return ch_score