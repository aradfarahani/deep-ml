def batch_requests(requests: list, max_batch_size: int, max_wait_time: float) -> list:
    """
    Group inference requests into batches based on size and time constraints.
    
    Args:
        requests: List of dicts with 'id', 'timestamp', 'features'
        max_batch_size: Maximum number of requests per batch
        max_wait_time: Maximum time to wait before processing a batch
    
    Returns:
        List of tuples: (request_ids, batched_features, process_time)
    """
    if not requests:
        return []
    
    # Sort requests by timestamp to process in chronological order
    sorted_requests = sorted(requests, key=lambda x: x['timestamp'])
    
    batches = []
    current_batch = []
    batch_start_time = None
    
    for req in sorted_requests:
        if not current_batch:
            # Start a new batch
            current_batch.append(req)
            batch_start_time = req['timestamp']
        else:
            time_elapsed = req['timestamp'] - batch_start_time
            
            # Check if we should finalize current batch
            if len(current_batch) >= max_batch_size or time_elapsed > max_wait_time:
                # Finalize current batch
                request_ids = [r['id'] for r in current_batch]
                batch_features = [r['features'] for r in current_batch]
                process_time = round(current_batch[-1]['timestamp'], 4)
                batches.append((request_ids, batch_features, process_time))
                
                # Start new batch with current request
                current_batch = [req]
                batch_start_time = req['timestamp']
            else:
                # Add to current batch
                current_batch.append(req)
    
    # Process remaining requests in the final batch
    if current_batch:
        request_ids = [r['id'] for r in current_batch]
        batch_features = [r['features'] for r in current_batch]
        process_time = round(current_batch[-1]['timestamp'], 4)
        batches.append((request_ids, batch_features, process_time))
    
    return batches