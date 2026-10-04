import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
	"""
	Implements Gaussian Naive Bayes classifier.
	
	Args:
		X_train: Training features (shape: N_train x D)
		y_train: Training labels (shape: N_train)
		X_test: Test features (shape: N_test x D)
	
	Returns:
		Predicted class labels for X_test (shape: N_test)
	"""
	classes = np.unique(y_train)
	n_classes = len(classes)
	n_features = X_train.shape[1]
	
	# Compute class priors, means, and variances
	priors = np.zeros(n_classes)
	means = np.zeros((n_classes, n_features))
	variances = np.zeros((n_classes, n_features))
	
	for idx, c in enumerate(classes):
		X_c = X_train[y_train == c]
		priors[idx] = X_c.shape[0] / X_train.shape[0]
		means[idx] = np.mean(X_c, axis=0)
		variances[idx] = np.var(X_c, axis=0)
	
	# Add small epsilon for numerical stability
	variances = variances + 1e-9
	
	# Predict for test samples
	predictions = []
	for x in X_test:
		posteriors = []
		for idx, c in enumerate(classes):
			# Log prior
			log_prior = np.log(priors[idx])
			# Log likelihood (Gaussian)
			log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variances[idx]) + 
										   (x - means[idx])**2 / variances[idx])
			posteriors.append(log_prior + log_likelihood)
		predictions.append(classes[np.argmax(posteriors)])
	
	return np.array(predictions)