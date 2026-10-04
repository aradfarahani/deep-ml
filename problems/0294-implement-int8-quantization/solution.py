import numpy as np

def int8_quantize(x: list[float]) -> dict:
	"""
	Perform symmetric INT8 quantization on a floating-point array.
	
	Args:
		x: Input list of floating-point values
		
	Returns:
		Dictionary with 'quantized', 'scale', and 'dequantized' keys
	"""
	arr = np.array(x, dtype=np.float32)
	
	# Find the maximum absolute value for symmetric quantization
	abs_max = np.max(np.abs(arr))
	
	# Calculate scale factor (avoid division by zero)
	if abs_max == 0:
		scale = 1.0
	else:
		scale = float(abs_max / 127.0)
	
	# Quantize: divide by scale, round, and clip to [-127, 127]
	quantized = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
	
	# Dequantize: multiply by scale
	dequantized = quantized.astype(np.float32) * scale
	
	return {
		'quantized': quantized.tolist(),
		'scale': round(scale, 6),
		'dequantized': [round(float(v), 4) for v in dequantized]
	}