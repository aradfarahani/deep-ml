def kv_cache_manager(num_layers: int, num_heads: int, head_dim: int, dtype_bytes: int, memory_budget_bytes: int, token_ids: list, token_scores: list, eviction_policy: str, num_protected: int = 0) -> dict:
    """
    Simulate a KV cache with memory budget and eviction policy.
    """
    # Calculate memory per token: 2 (K and V) * layers * heads * head_dim * dtype_bytes
    bytes_per_token = 2 * num_layers * num_heads * head_dim * dtype_bytes
    
    # Calculate max tokens that fit in memory budget
    max_tokens = int(memory_budget_bytes // bytes_per_token)
    
    # Simulate KV cache with eviction
    cache = []  # list of dicts: {token_id, score, position}
    evicted = []
    
    for i, (tid, score) in enumerate(zip(token_ids, token_scores)):
        # Check if cache is full before adding
        if len(cache) >= max_tokens:
            # Get non-protected candidates (position >= num_protected)
            candidates = [(idx, entry) for idx, entry in enumerate(cache) if entry['position'] >= num_protected]
            
            if candidates:
                if eviction_policy == 'fifo':
                    # Evict oldest non-protected token
                    evict_idx = min(candidates, key=lambda x: x[1]['position'])[0]
                elif eviction_policy == 'score':
                    # Evict lowest score; break ties by oldest (smallest position)
                    evict_idx = min(candidates, key=lambda x: (x[1]['score'], x[1]['position']))[0]
                else:
                    raise ValueError(f"Unknown policy: {eviction_policy}")
                
                evicted.append(cache.pop(evict_idx)['token_id'])
        
        # Add new token if there is space
        if len(cache) < max_tokens:
            cache.append({'token_id': tid, 'score': score, 'position': i})
    
    final_cache = [entry['token_id'] for entry in cache]
    
    return {
        'bytes_per_token': bytes_per_token,
        'max_tokens': max_tokens,
        'final_cache': final_cache,
        'evicted_tokens': evicted,
        'num_evictions': len(evicted)
    }