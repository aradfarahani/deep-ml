def autoscale_simulate(rps_series: list, capacity_per_replica: int, min_replicas: int, max_replicas: int, scale_up_threshold: float, scale_down_threshold: float, cooldown_steps: int) -> dict:
	current_replicas = min_replicas
	cooldown_counter = 0
	total_violations = 0
	total_dropped = 0
	max_replicas_used = current_replicas
	utilizations = []

	for rps in rps_series:
		# Compute current capacity and utilization
		capacity = current_replicas * capacity_per_replica
		utilization = rps / capacity
		utilizations.append(utilization)

		# Check SLA
		if rps > capacity:
			total_violations += 1
			total_dropped += rps - capacity

		# Scaling decision
		if cooldown_counter == 0:
			if utilization > scale_up_threshold:
				new_replicas = min(current_replicas + 1, max_replicas)
				if new_replicas != current_replicas:
					current_replicas = new_replicas
					cooldown_counter = cooldown_steps
			elif utilization < scale_down_threshold:
				new_replicas = max(current_replicas - 1, min_replicas)
				if new_replicas != current_replicas:
					current_replicas = new_replicas
					cooldown_counter = cooldown_steps
		else:
			cooldown_counter -= 1

		max_replicas_used = max(max_replicas_used, current_replicas)

	avg_util = round(sum(utilizations) / len(utilizations), 2)

	return {
		'final_replicas': current_replicas,
		'total_sla_violations': total_violations,
		'average_utilization': avg_util,
		'max_replicas_used': max_replicas_used,
		'total_dropped_requests': total_dropped
	}