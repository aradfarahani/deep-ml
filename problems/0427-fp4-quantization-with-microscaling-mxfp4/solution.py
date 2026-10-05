import numpy as np

def mxfp4_quantize(x: list, block_size: int = 4) -> dict:
	x_arr = np.array(x, dtype=np.float64).flatten()
	n = len(x_arr)

	# Pad to multiple of block_size
	pad_len = (block_size - n % block_size) % block_size
	if pad_len > 0:
		x_arr = np.concatenate([x_arr, np.zeros(pad_len)])

	# FP4 E2M1 representable values (positive)
	fp4_pos = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
	# Build full set including negatives, sorted
	fp4_all = sorted([-v for v in fp4_pos if v > 0] + fp4_pos)
	fp4_all = np.array(fp4_all)

	blocks = x_arr.reshape(-1, block_size)
	result = np.zeros_like(blocks)
	scales = []

	for i, block in enumerate(blocks):
		amax = np.max(np.abs(block))

		if amax == 0:
			scale = 1.0
		else:
			raw_scale = amax / 6.0
			scale = float(2.0 ** np.ceil(np.log2(raw_scale)))

		scales.append(scale)

		for j in range(block_size):
			val = block[j] / scale
			# Find nearest FP4 value, ties broken by smaller absolute value
			dists = np.abs(fp4_all - val)
			min_dist = np.min(dists)
			candidates = np.where(np.abs(dists - min_dist) < 1e-12)[0]
			# Among tied candidates, pick one with smallest absolute FP4 value
			best_idx = candidates[np.argmin(np.abs(fp4_all[candidates]))]
			result[i, j] = fp4_all[best_idx] * scale

	quantized = result.flatten()[:n].tolist()
	quantized = [round(v, 4) for v in quantized]
	scales_rounded = [round(s, 4) for s in scales]

	return {"quantized": quantized, "scales": scales_rounded}