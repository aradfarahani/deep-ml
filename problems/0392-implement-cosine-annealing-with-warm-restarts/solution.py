import numpy as np

def cosine_annealing_warm_restarts(eta_max: float, eta_min: float, T_0: int, T_mult: int, total_epochs: int) -> list:
	"""
	Compute learning rates using cosine annealing with warm restarts.
	
	Args:
		eta_max: Maximum learning rate
		eta_min: Minimum learning rate
		T_0: Number of epochs in first cycle
		T_mult: Cycle length multiplier after each restart
		total_epochs: Total number of epochs
	
	Returns:
		List of learning rates for each epoch, rounded to 4 decimal places
	"""
	lrs = []
	for epoch in range(total_epochs):
		T_i = T_0
		T_cur = epoch
		
		if T_mult == 1:
			T_cur = epoch % T_0
		else:
			while T_cur >= T_i:
				T_cur -= T_i
				T_i = T_i * T_mult
		
		lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * T_cur / T_i))
		lrs.append(round(float(lr), 4))
	
	return lrs