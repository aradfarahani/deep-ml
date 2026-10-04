import numpy as np

def square_relu(x: np.ndarray) -> dict:
	"""
	Apply the Square ReLU activation function and compute its derivative.
	
	Args:
		x: Input numpy array of any shape
	
	Returns:
		Dictionary with 'output' and 'derivative' as numpy arrays
	"""
	x = np.array(x, dtype=float)
	
	# Square ReLU: f(x) = x^2 if x > 0, else 0
	mask = x > 0
	output = np.where(mask, x ** 2, 0.0)
	
	# Derivative: f'(x) = 2x if x > 0, else 0
	derivative = np.where(mask, 2 * x, 0.0)
	
	return {
		'output': np.round(output, 4),
		'derivative': np.round(derivative, 4)
	}