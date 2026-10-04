import numpy as np

def davies_bouldin_index(X, labels):
    """
    Calculate the Davies-Bouldin Index for clustering evaluation.
    
    Parameters:
    X: numpy array of shape (n_samples, n_features) - data points
    labels: numpy array of shape (n_samples,) - cluster labels
    
    Returns:
    float: Davies-Bouldin Index rounded to 4 decimal places
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return 0.0
    
    # Calculate centroids and scatter for each cluster
    centroids = []
    scatters = []
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
        
        # Scatter: average Euclidean distance from points to centroid
        distances = np.sqrt(np.sum((cluster_points - centroid) ** 2, axis=1))
        scatter = np.mean(distances)
        scatters.append(scatter)
    
    centroids = np.array(centroids)
    scatters = np.array(scatters)
    
    # Calculate R values for each cluster
    R = []
    for i in range(n_clusters):
        max_r = 0
        for j in range(n_clusters):
            if i != j:
                d_ij = np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
                if d_ij > 0:
                    r_ij = (scatters[i] + scatters[j]) / d_ij
                    max_r = max(max_r, r_ij)
        R.append(max_r)
    
    dbi = np.mean(R)
    return round(dbi, 4)