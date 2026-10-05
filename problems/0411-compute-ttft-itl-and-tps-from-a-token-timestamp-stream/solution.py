def compute_inference_metrics(timestamps: list[float]) -> dict:
	"""
	Compute LLM inference performance metrics from token timestamps.
	
	Args:
		timestamps: List of floats where timestamps[0] is the request start time
		            and timestamps[1:] are the times when each output token was generated.
	
	Returns:
		Dictionary with keys 'ttft', 'tps', 'itl' containing the metric values.
	"""
	n = len(timestamps)
	num_tokens = n - 1
	
	# Time To First Token
	ttft = timestamps[1] - timestamps[0]
	
	# Inter-Token Latency
	if num_tokens > 1:
		inter_token_latencies = [timestamps[i + 1] - timestamps[i] for i in range(1, n - 1)]
		itl = sum(inter_token_latencies) / len(inter_token_latencies)
	else:
		itl = 0.0
	
	# Tokens Per Second
	total_time = timestamps[-1] - timestamps[0]
	if total_time > 0:
		tps = num_tokens / total_time
	else:
		tps = 0.0
	
	return {"ttft": ttft, "tps": tps, "itl": itl}