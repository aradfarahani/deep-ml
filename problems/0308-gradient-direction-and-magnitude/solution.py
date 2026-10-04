import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	grad = np.array(gradient, dtype=float)
	
	# Calculate magnitude (L2 norm)
	magnitude = np.linalg.norm(grad)
	
	# Handle zero gradient case
	if magnitude < 1e-10:
		direction = np.zeros_like(grad)
		descent_direction = np.zeros_like(grad)
	else:
		# Direction is the unit vector (steepest ascent)
		direction = grad / magnitude
		# Steepest descent is negative of gradient direction
		descent_direction = -direction
	
	return {
		'magnitude': float(magnitude),
		'direction': direction.tolist(),
		'descent_direction': descent_direction.tolist()
	}