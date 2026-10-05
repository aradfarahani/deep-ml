def cache_aware_route(replicas: list, requests: list, alpha: float = 0.7, beta: float = 0.3) -> dict:
    # Copy replica states to avoid mutating input
    replica_states = []
    for r in replicas:
        replica_states.append({
            'cached_prefixes': [list(p) for p in r['cached_prefixes']],
            'active_requests': r['active_requests'],
            'max_capacity': r['max_capacity']
        })
    
    assignments = []
    cache_hit_rates = []
    
    def longest_prefix_match(tokens, prefixes):
        max_match = 0
        for prefix in prefixes:
            match = 0
            for a, b in zip(tokens, prefix):
                if a == b:
                    match += 1
                else:
                    break
            max_match = max(max_match, match)
        return max_match
    
    for req in requests:
        tokens = req['tokens']
        req_len = len(tokens)
        best_replica = -1
        best_score = float('-inf')
        best_hit_rate = 0.0
        
        for i, rep in enumerate(replica_states):
            if rep['active_requests'] >= rep['max_capacity']:
                continue
            
            match_len = longest_prefix_match(tokens, rep['cached_prefixes'])
            hit_rate = match_len / req_len if req_len > 0 else 0.0
            load_ratio = rep['active_requests'] / rep['max_capacity']
            score = alpha * hit_rate - beta * load_ratio
            
            if score > best_score:
                best_score = score
                best_replica = i
                best_hit_rate = hit_rate
        
        # All replicas full: pick lowest load ratio
        if best_replica == -1:
            min_load = float('inf')
            for i, rep in enumerate(replica_states):
                load_ratio = rep['active_requests'] / rep['max_capacity']
                if load_ratio < min_load:
                    min_load = load_ratio
                    best_replica = i
            match_len = longest_prefix_match(tokens, replica_states[best_replica]['cached_prefixes'])
            best_hit_rate = match_len / req_len if req_len > 0 else 0.0
        
        assignments.append(best_replica)
        cache_hit_rates.append(round(best_hit_rate, 2))
        replica_states[best_replica]['active_requests'] += 1
    
    avg_hit = round(sum(cache_hit_rates) / len(cache_hit_rates), 2) if cache_hit_rates else 0.0
    load_dist = [rep['active_requests'] for rep in replica_states]
    
    return {
        'assignments': assignments,
        'cache_hit_rates': cache_hit_rates,
        'avg_cache_hit_rate': avg_hit,
        'load_distribution': load_dist
    }