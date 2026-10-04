import numpy as np

def agglomerative_clustering(X: list, n_clusters: int, linkage: str = 'single') -> list:
    """
    Perform agglomerative hierarchical clustering.
    
    Args:
        X: Data points, shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        linkage: 'single', 'complete', or 'average'
    
    Returns:
        List of cluster labels for each sample
    """
    X = np.array(X, dtype=float)
    n_samples = X.shape[0]
    
    # Initialize: each point is its own cluster
    clusters = {i: [i] for i in range(n_samples)}
    
    def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    def cluster_distance(c1, c2):
        points1 = [X[i] for i in clusters[c1]]
        points2 = [X[i] for i in clusters[c2]]
        
        distances = []
        for p1 in points1:
            for p2 in points2:
                distances.append(euclidean_distance(p1, p2))
        
        if linkage == 'single':
            return min(distances)
        elif linkage == 'complete':
            return max(distances)
        elif linkage == 'average':
            return sum(distances) / len(distances)
    
    while len(clusters) > n_clusters:
        # Find the two closest clusters
        min_dist = float('inf')
        merge_pair = None
        
        cluster_ids = sorted(clusters.keys())
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                c1, c2 = cluster_ids[i], cluster_ids[j]
                dist = cluster_distance(c1, c2)
                if dist < min_dist:
                    min_dist = dist
                    merge_pair = (c1, c2)
        
        # Merge the two closest clusters (merge into the one with smaller id)
        c1, c2 = merge_pair
        clusters[c1] = clusters[c1] + clusters[c2]
        del clusters[c2]
    
    # Assign labels based on sorted cluster IDs
    labels = [0] * n_samples
    cluster_ids = sorted(clusters.keys())
    for label, cluster_id in enumerate(cluster_ids):
        for member in clusters[cluster_id]:
            labels[member] = label
    
    return labels