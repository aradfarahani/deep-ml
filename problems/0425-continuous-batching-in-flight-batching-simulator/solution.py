def continuous_batching_sim(requests: list[dict], max_batch_size: int) -> dict:
    """
    Simulate continuous batching for LLM inference.
    """
    n = len(requests)
    if n == 0:
        return {'total_time': 0, 'avg_latency': 0.0, 'avg_ttft': 0.0, 'throughput': 0.0}
    
    # Create indexed list sorted by arrival_time, then original order
    indexed = []
    for i, r in enumerate(requests):
        indexed.append((r['arrival_time'], i, r['tokens_needed']))
    indexed.sort(key=lambda x: (x[0], x[1]))
    
    queue = []          # (tokens_needed, original_index, arrival_time)
    active = []         # [tokens_remaining, original_index, arrival_time, start_time]
    completed = []      # (original_index, arrival_time, start_time, end_time)
    
    next_idx = 0        # pointer into indexed
    time = 0
    
    while len(completed) < n:
        # 1. Arrive: add requests with arrival_time <= time
        while next_idx < n and indexed[next_idx][0] <= time:
            arr_time, orig_idx, tok_needed = indexed[next_idx]
            queue.append((tok_needed, orig_idx, arr_time))
            next_idx += 1
        
        # 2. Fill empty slots from queue
        while len(active) < max_batch_size and queue:
            tok_needed, orig_idx, arr_time = queue.pop(0)
            active.append([tok_needed, orig_idx, arr_time, time])
        
        # 3. Generate one token per active slot
        for slot in active:
            slot[0] -= 1
        
        # 4. Complete: remove finished sequences
        still_active = []
        for slot in active:
            if slot[0] == 0:
                completed.append((slot[1], slot[2], slot[3], time))
            else:
                still_active.append(slot)
        active = still_active
        
        # 5. Advance time
        if not active and not queue and next_idx < n:
            time = indexed[next_idx][0]
        else:
            time += 1
    
    total_tokens = sum(r['tokens_needed'] for r in requests)
    total_time = time
    
    latencies = [end - arr for _, arr, _, end in completed]
    ttfts = [start - arr for _, arr, start, _ in completed]
    
    avg_latency = sum(latencies) / len(latencies)
    avg_ttft = sum(ttfts) / len(ttfts)
    throughput = total_tokens / total_time
    
    return {
        'total_time': total_time,
        'avg_latency': round(avg_latency, 4),
        'avg_ttft': round(avg_ttft, 4),
        'throughput': round(throughput, 4)
    }