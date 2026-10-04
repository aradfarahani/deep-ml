import numpy as np

def dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Implement DBSCAN clustering algorithm.
    
    Parameters:
    - X: 2D numpy array of shape (n_samples, n_features)
    - eps: Maximum distance between two samples to be considered neighbors
    - min_samples: Minimum number of samples in a neighborhood for a core point
    
    Returns:
    - labels: 1D numpy array of cluster labels (-1 for noise points)
    """
    n_samples = X.shape[0]
    labels = np.full(n_samples, -1)
    visited = np.zeros(n_samples, dtype=bool)
    cluster_id = 0
    
    def get_neighbors(point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return list(np.where(distances <= eps)[0])
    
    for i in range(n_samples):
        if visited[i]:
            continue
        
        visited[i] = True
        neighbors = get_neighbors(i)
        
        if len(neighbors) < min_samples:
            continue
        
        labels[i] = cluster_id
        seed_set = list(set(neighbors) - {i})
        
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            
            if not visited[q]:
                visited[q] = True
                q_neighbors = get_neighbors(q)
                
                if len(q_neighbors) >= min_samples:
                    for n in q_neighbors:
                        if n not in seed_set:
                            seed_set.append(n)
            
            if labels[q] == -1:
                labels[q] = cluster_id
            
            j += 1
        
        cluster_id += 1
    
    return labels