import numpy as np

def transformer_block(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray, mode: str, eps: float = 1e-5) -> np.ndarray:
	"""
	Apply a transformer block with two sublayers using either pre-norm or post-norm.
	
	Args:
		x: Input array of shape (seq_len, d_model)
		W1, b1: Weights and bias for first sublayer
		W2, b2: Weights and bias for second sublayer
		gamma1, beta1: LayerNorm params for first normalization
		gamma2, beta2: LayerNorm params for second normalization
		mode: 'pre_norm' or 'post_norm'
		eps: Epsilon for numerical stability
	
	Returns:
		Output array of shape (seq_len, d_model)
	"""
	def layer_norm(z, gamma, beta):
		mean = np.mean(z, axis=-1, keepdims=True)
		var = np.var(z, axis=-1, keepdims=True)
		z_norm = (z - mean) / np.sqrt(var + eps)
		return gamma * z_norm + beta
	
	def sublayer_fn(z, W, b):
		return z @ W + b
	
	if mode == "pre_norm":
		# Sublayer 1: LN before sublayer, then residual
		h = x + sublayer_fn(layer_norm(x, gamma1, beta1), W1, b1)
		# Sublayer 2: LN before sublayer, then residual
		out = h + sublayer_fn(layer_norm(h, gamma2, beta2), W2, b2)
	elif mode == "post_norm":
		# Sublayer 1: residual then LN
		h = layer_norm(x + sublayer_fn(x, W1, b1), gamma1, beta1)
		# Sublayer 2: residual then LN
		out = layer_norm(h + sublayer_fn(h, W2, b2), gamma2, beta2)
	else:
		raise ValueError("mode must be 'pre_norm' or 'post_norm'")
	
	return out