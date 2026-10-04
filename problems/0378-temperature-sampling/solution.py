import numpy as np

def temperature_sampling(logits: np.ndarray, temperature: float) -> list:
	"""
	Compute temperature-scaled softmax probabilities from logits.
	
	Args:
		logits: 1D numpy array of raw model output scores
		temperature: float controlling distribution sharpness
	
	Returns:
		List of probabilities after temperature scaling
	"""
	if temperature <= 0:
		probs = np.zeros(len(logits), dtype=float)
		probs[np.argmax(logits)] = 1.0
		return probs.tolist()
	
	scaled_logits = logits / temperature
	# Subtract max for numerical stability
	scaled_logits = scaled_logits - np.max(scaled_logits)
	exp_logits = np.exp(scaled_logits)
	probs = exp_logits / np.sum(exp_logits)
	
	return probs.tolist()