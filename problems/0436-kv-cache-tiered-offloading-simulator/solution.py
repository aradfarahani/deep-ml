def kv_cache_offload_sim(tiers: list, entries: list, recompute_latency_us: float = 1000.0) -> dict:
	# Sort tiers by latency ascending (fastest first)
	sorted_tiers = sorted(tiers, key=lambda t: t['latency_us'])

	# Sort entries by recency descending (most recent first)
	sorted_entries = sorted(entries, key=lambda e: e['recency'], reverse=True)

	# Initialize placement and remaining capacity
	placement = {t['name']: [] for t in sorted_tiers}
	remaining_capacity = {t['name']: t['capacity'] for t in sorted_tiers}
	evicted = []
	entry_tier = {}

	# Greedily assign entries to fastest available tier
	for entry in sorted_entries:
		placed = False
		for tier in sorted_tiers:
			if remaining_capacity[tier['name']] > 0:
				placement[tier['name']].append(entry['id'])
				remaining_capacity[tier['name']] -= 1
				entry_tier[entry['id']] = tier['name']
				placed = True
				break
		if not placed:
			evicted.append(entry['id'])
			entry_tier[entry['id']] = 'evicted'

	# Build tier latency lookup
	tier_latency = {t['name']: t['latency_us'] for t in sorted_tiers}

	# Compute total latency (each entry accessed once)
	total_latency = 0.0
	for entry in entries:
		if entry_tier[entry['id']] == 'evicted':
			total_latency += recompute_latency_us
		else:
			total_latency += tier_latency[entry_tier[entry['id']]]

	n = len(entries)
	avg_latency = total_latency / n if n > 0 else 0.0

	# Sort ids within each tier and evicted list
	for key in placement:
		placement[key] = sorted(placement[key])
	evicted = sorted(evicted)

	return {
		'placement': placement,
		'evicted': evicted,
		'total_latency_us': round(total_latency, 2),
		'avg_latency_us': round(avg_latency, 2)
	}