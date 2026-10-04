import math

def mish(x: float) -> float:
	"""
	Compute the Mish activation function.

	Args:
		x (float): Input value

	Returns:
		float: Mish activation value rounded to 4 decimal places
	"""
	softplus = math.log(1 + math.exp(x))
	return round(x * math.tanh(softplus), 4)