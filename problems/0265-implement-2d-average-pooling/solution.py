import numpy as np

def avg_pool_2d(input_matrix: list[list[float]], pool_size: int) -> list[list[float]]:
	"""
	Perform 2D average pooling on the input matrix.
	
	Args:
		input_matrix: 2D input array of shape (H, W)
		pool_size: Size of the square pooling window
		
	Returns:
		2D array after average pooling of shape (H//pool_size, W//pool_size)
	"""
	X = np.array(input_matrix, dtype=float)
	H, W = X.shape
	
	out_h = H // pool_size
	out_w = W // pool_size
	
	output = np.zeros((out_h, out_w))
	
	for i in range(out_h):
		for j in range(out_w):
			region = X[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
			output[i, j] = np.mean(region)
	
	return output.tolist()