import numpy as np

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
	"""
	Calculate the Silhouette Score for clustering evaluation.
	
	X: shape (n_samples, n_features) - data points
	labels: shape (n_samples,) - cluster label for each point
	
	Returns: silhouette score (float between -1 and 1)
	"""
	n_samples = X.shape[0]
	unique_labels = np.unique(labels)
	n_clusters = len(unique_labels)
	
	# Edge cases
	if n_clusters == 1 or n_clusters == n_samples:
		return 0.0
	
	silhouette_vals = []
	
	for i in range(n_samples):
		current_label = labels[i]
		
		# Calculate a(i): mean intra-cluster distance
		same_cluster_mask = labels == current_label
		same_cluster_points = X[same_cluster_mask]
		
		if np.sum(same_cluster_mask) == 1:
			a_i = 0.0
		else:
			distances = np.linalg.norm(same_cluster_points - X[i], axis=1)
			a_i = np.sum(distances) / (np.sum(same_cluster_mask) - 1)
		
		# Calculate b(i): minimum mean inter-cluster distance
		b_i = np.inf
		for label in unique_labels:
			if label != current_label:
				other_cluster_points = X[labels == label]
				distances = np.linalg.norm(other_cluster_points - X[i], axis=1)
				mean_dist = np.mean(distances)
				b_i = min(b_i, mean_dist)
		
		# Calculate silhouette coefficient for point i
		if max(a_i, b_i) == 0:
			s_i = 0.0
		else:
			s_i = (b_i - a_i) / max(a_i, b_i)
		
		silhouette_vals.append(s_i)
	
	return round(np.mean(silhouette_vals), 4)