from typing import List, Tuple
import math

def train_paris_model(
    features: List[List[float]],
    targets: List[float],
    cluster_ids: List[int],
    n_clusters: int
) -> Tuple[List[float], List[List[float]]]:
    """
    Train Paris-style decentralized model.
    
    1. Partition data by cluster_ids (simulating DINOv2 semantic clustering)
    2. Train each expert on its partition IN ISOLATION (learns mean target)
    3. Train router by computing cluster centroids (learns to route by similarity)
    
    NO communication between experts during training!
    
    Args:
        features: N x D feature vectors
        targets: N target values
        cluster_ids: Cluster assignment for each data point
        n_clusters: Number of expert clusters
    
    Returns:
        expert_predictions: Mean target learned by each expert
        router_centroids: Mean feature vector for each cluster
    """
    # Partition data by cluster assignment
    partitions = [[] for _ in range(n_clusters)]
    for i, cid in enumerate(cluster_ids):
        partitions[cid].append((features[i], targets[i]))
    
    # Train each expert in COMPLETE ISOLATION
    expert_predictions = []
    for partition in partitions:
        if partition:
            mean_target = sum(t for f, t in partition) / len(partition)
        else:
            mean_target = 0.0
        expert_predictions.append(mean_target)
    
    # Train router: compute centroid of each cluster's features
    router_centroids = []
    dim = len(features[0]) if features else 0
    for partition in partitions:
        if partition:
            centroid = [0.0] * dim
            for f, t in partition:
                for d in range(dim):
                    centroid[d] += f[d]
            centroid = [c / len(partition) for c in centroid]
        else:
            centroid = [0.0] * dim
        router_centroids.append(centroid)
    
    return expert_predictions, router_centroids


def paris_inference(
    feature: List[float],
    expert_predictions: List[float],
    router_centroids: List[List[float]],
    strategy: str = 'top1'
) -> float:
    """
    Run inference with trained Paris model.
    
    Router selects expert(s) based on feature similarity to learned centroids.
    
    Args:
        feature: Input feature vector
        expert_predictions: Trained expert outputs
        router_centroids: Learned cluster centroids for routing
        strategy: 'top1' (select nearest) or 'weighted' (inverse distance weighting)
    
    Returns:
        Model prediction
    """
    def euclidean_distance(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    distances = [euclidean_distance(feature, c) for c in router_centroids]
    
    if strategy == 'top1':
        best_idx = distances.index(min(distances))
        return round(expert_predictions[best_idx], 2)
    else:  # weighted
        eps = 1e-8
        inv_distances = [1.0 / (d + eps) for d in distances]
        total = sum(inv_distances)
        weights = [w / total for w in inv_distances]
        prediction = sum(w * p for w, p in zip(weights, expert_predictions))
        return round(prediction, 2)