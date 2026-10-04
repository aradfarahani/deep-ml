import numpy as np

def kmeans_plus_plus_init(X: np.ndarray, k: int, seed: int = None) -> np.ndarray:
	"""
	Initialize k centroids using the K-Means++ algorithm.
	
	Args:
		X: Data points of shape (n_samples, n_features)
		k: Number of centroids to initialize
		seed: Random seed for reproducibility
	
	Returns:
		Centroids of shape (k, n_features)
	"""
	if seed is not None:
		np.random.seed(seed)
	
	n_samples = X.shape[0]
	centroids = []
	
	# Step 1: Choose the first centroid uniformly at random
	first_idx = np.random.randint(0, n_samples)
	centroids.append(X[first_idx].copy())
	
	# Step 2: Choose remaining k-1 centroids
	for _ in range(1, k):
		# Compute squared distance from each point to nearest centroid
		distances_squared = np.array([
			min(np.sum((x - c) ** 2) for c in centroids)
			for x in X
		])
		
		# Compute selection probabilities (proportional to D^2)
		probabilities = distances_squared / distances_squared.sum()
		
		# Sample next centroid
		next_idx = np.random.choice(n_samples, p=probabilities)
		centroids.append(X[next_idx].copy())
	
	return np.array(centroids)