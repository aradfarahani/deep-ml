import numpy as np

def mini_batch_kmeans(X: np.ndarray, k: int, batch_size: int, max_iters: int, seed: int = 42) -> list:
    """
    Perform Mini-Batch K-Means clustering.
    
    Args:
        X: Data points of shape (n_samples, n_features)
        k: Number of clusters
        batch_size: Size of each mini-batch
        max_iters: Number of iterations
        seed: Random seed for reproducibility
    
    Returns:
        List of k centroids, each centroid is a list of coordinates
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    
    # Initialize centroids randomly from data points
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices].astype(float).copy()
    
    # Count how many times each centroid has been updated
    counts = np.zeros(k)
    
    for _ in range(max_iters):
        # Sample a mini-batch
        batch_indices = np.random.choice(n_samples, batch_size, replace=False)
        batch = X[batch_indices]
        
        # Assign each point in batch to nearest centroid
        assignments = []
        for point in batch:
            distances = np.sum((centroids - point) ** 2, axis=1)
            closest = np.argmin(distances)
            assignments.append(closest)
        
        # Update centroids using streaming gradient descent
        for i, point in enumerate(batch):
            c = assignments[i]
            counts[c] += 1
            eta = 1.0 / counts[c]
            centroids[c] = (1 - eta) * centroids[c] + eta * point
    
    return np.round(centroids, 4).tolist()