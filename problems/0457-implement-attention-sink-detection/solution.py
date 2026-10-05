import numpy as np

def detect_attention_sinks(attn_weights: np.ndarray, threshold: float) -> dict:
	"""
	Detect attention sink tokens from multi-head attention weight matrices.
	
	Args:
		attn_weights: Attention weights of shape (num_heads, seq_len, seq_len)
		threshold: Minimum average received attention to qualify as a sink
		
	Returns:
		Dictionary with 'sink_positions', 'avg_attention_received', and 'sink_scores'
	"""
	# Step 1: Average attention over all heads -> shape (seq_len, seq_len)
	avg_over_heads = np.mean(attn_weights, axis=0)
	
	# Step 2: For each key position j, compute mean attention received
	# across all query positions i -> shape (seq_len,)
	avg_attention_received = np.mean(avg_over_heads, axis=0)
	
	# Step 3: Identify sink positions (those >= threshold)
	sink_positions = sorted(np.where(avg_attention_received >= threshold)[0].tolist())
	
	# Step 4: Extract sink scores
	sink_scores = [round(float(avg_attention_received[p]), 4) for p in sink_positions]
	
	# Step 5: Round all values for clean output
	avg_attention_received_rounded = [round(float(x), 4) for x in avg_attention_received]
	
	return {
		'sink_positions': sink_positions,
		'avg_attention_received': avg_attention_received_rounded,
		'sink_scores': sink_scores
	}