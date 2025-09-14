import numpy as np

def muonclip_qk_clip(W_q: np.ndarray, W_k: np.ndarray, x: np.ndarray, t: float, alpha: float = 0.5, eps: float = 1e-7):
	# Ensure arrays
	W_q = np.array(W_q, dtype=float)
	W_k = np.array(W_k, dtype=float)
	x   = np.array(x,   dtype=float)

	# Dimensions
	d_head = W_q.shape[0]

	# q, k and QK scores with 1/sqrt(d_head) scaling
	q = np.einsum('bsd,hd->bsh', x, W_q)
	k = np.einsum('bsd,hd->bsh', x, W_k)
	scores = np.einsum('bih,bjh->bij', q, k) / np.sqrt(max(d_head, 1))
	max_pre = float(scores.max())

	W_q_new, W_k_new = W_q.copy(), W_k.copy()
	clipped = False
	if max_pre > t:
		eta = t / (max_pre + eps)
		scale_q = eta ** alpha
		scale_k = eta ** (1 - alpha)
		W_q_new = W_q_new * scale_q
		W_k_new = W_k_new * scale_k
		clipped = True

	# Recompute post-clip max for reporting
	q_post = np.einsum('bsd,hd->bsh', x, W_q_new)
	k_post = np.einsum('bsd,hd->bsh', x, W_k_new)
	scores_post = np.einsum('bih,bjh->bij', q_post, k_post) / np.sqrt(max(d_head, 1))
	max_post = float(s