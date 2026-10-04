import numpy as np

def soft_voting_classifier(probabilities: np.ndarray, weights: list = None) -> list:
	"""
	Implement soft voting for ensemble classification.
	
	Args:
		probabilities: 3D array of shape (n_classifiers, n_samples, n_classes)
		weights: Optional list of weights for each classifier
	
	Returns:
		List of predicted class labels for each sample
	"""
	n_classifiers = probabilities.shape[0]
	
	if weights is None:
		weights = np.ones(n_classifiers)
	else:
		weights = np.array(weights)
	
	# Normalize weights to sum to 1
	weights = weights / np.sum(weights)
	
	# Compute weighted average of probabilities across classifiers
	weighted_probs = np.average(probabilities, axis=0, weights=weights)
	
	# Predict class with highest average probability
	predictions = np.argmax(weighted_probs, axis=1)
	
	return predictions.tolist()