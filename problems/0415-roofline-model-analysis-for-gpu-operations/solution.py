def roofline_analysis(peak_gflops: float, peak_bandwidth_gbs: float, operations: list) -> dict:
	"""
	Perform Roofline Model analysis for GPU operations.

	Args:
		peak_gflops: Peak compute throughput in GFLOPS
		peak_bandwidth_gbs: Peak memory bandwidth in GB/s
		operations: List of dicts with keys 'name', 'flops', 'bytes'

	Returns:
		Dict with 'ridge_point' and 'operations' list containing
		per-operation analysis results.
	"""
	# Ridge point: the operational intensity where the memory bandwidth
	# ceiling meets the compute ceiling
	ridge_point = peak_gflops / peak_bandwidth_gbs

	results = []
	for op in operations:
		# Operational intensity in FLOP/byte
		oi = op['flops'] / op['bytes']

		# Attainable performance is the minimum of the compute ceiling
		# and the bandwidth ceiling scaled by operational intensity
		attainable = min(peak_gflops, peak_bandwidth_gbs * oi)

		# Classification: at or above the ridge point means compute-bound
		bottleneck = 'compute-bound' if oi >= ridge_point else 'memory-bound'

		# Efficiency as percentage of peak compute
		efficiency = (attainable / peak_gflops) * 100.0

		results.append({
			'name': op['name'],
			'operational_intensity': oi,
			'attainable_gflops': attainable,
			'bottleneck': bottleneck,
			'efficiency': efficiency
		})

	return {
		'ridge_point': ridge_point,
		'operations': results
	}