import numpy as np

def lle(X: np.ndarray, n_neighbors: int, n_components: int) -> np.ndarray:
	"""
	Perform Locally Linear Embedding for dimensionality reduction.
	
	Args:
		X: Data matrix of shape (n_samples, n_features)
		n_neighbors: Number of nearest neighbors to use
		n_components: Target dimensionality
	
	Returns:
		Embedding Y of shape (n_samples, n_components)
	"""
	n_samples = X.shape[0]
	
	# Step 1: Compute pairwise squared Euclidean distances
	distances = np.zeros((n_samples, n_samples))
	for i in range(n_samples):
		for j in range(n_samples):
			distances[i, j] = np.sum((X[i] - X[j]) ** 2)
	
	# Find k nearest neighbors (excluding self)
	neighbors = np.zeros((n_samples, n_neighbors), dtype=int)
	for i in range(n_samples):
		sorted_idx = np.argsort(distances[i])
		neighbors[i] = sorted_idx[1:n_neighbors+1]
	
	# Step 2: Compute reconstruction weights
	W = np.zeros((n_samples, n_samples))
	reg = 1e-3
	
	for i in range(n_samples):
		nbr_idx = neighbors[i]
		Z = X[nbr_idx] - X[i]
		C = Z @ Z.T
		C += reg * np.eye(n_neighbors)
		w = np.linalg.solve(C, np.ones(n_neighbors))
		w = w / np.sum(w)
		W[i, nbr_idx] = w
	
	# Step 3: Build cost matrix M = (I - W)^T (I - W)
	I_W = np.eye(n_samples) - W
	M = I_W.T @ I_W
	
	# Step 4: Eigendecomposition - find smallest non-zero eigenvalues
	eigenvalues, eigenvectors = np.linalg.eigh(M)
	
	# Take eigenvectors corresponding to smallest non-zero eigenvalues
	# Skip the first eigenvector (eigenvalue ~0)
	Y = eigenvectors[:, 1:n_components+1]
	
	# Standardize sign: make sum of each column positive
	for j in range(n_components):
		if np.sum(Y[:, j]) < 0:
			Y[:, j] *= -1
	
	return np.round(Y, 4)