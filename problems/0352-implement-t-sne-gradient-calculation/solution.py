import numpy as np

def tsne_gradient(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
	"""
	Compute the gradient of the t-SNE cost function.
	
	Args:
		P: (n, n) symmetric numpy array of joint probabilities in high-dimensional space
		Y: (n, d) numpy array of current low-dimensional embedding
	
	Returns:
		gradient: (n, d) numpy array of gradients for each point, rounded to 4 decimal places
	"""
	n = Y.shape[0]
	
	# Compute pairwise differences: diff[i,j] = Y[i] - Y[j]
	diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
	
	# Compute pairwise squared Euclidean distances
	D = np.sum(diff ** 2, axis=2)
	
	# Compute Q distribution using Student's t-distribution (df=1)
	inv_dist = 1.0 / (1.0 + D)
	np.fill_diagonal(inv_dist, 0.0)
	Q = inv_dist / np.sum(inv_dist)
	Q = np.maximum(Q, 1e-12)
	
	# Compute gradient
	PQ = P - Q
	weights = (PQ * inv_dist)[:, :, np.newaxis]
	
	grad = 4.0 * np.sum(weights * diff, axis=1)
	
	return np.round(grad, 4)