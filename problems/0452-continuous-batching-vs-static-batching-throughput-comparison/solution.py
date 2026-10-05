def compare_batching(requests: list, max_batch_size: int, time_per_step: float) -> dict:
    n = len(requests)
    total_tokens = sum(requests)
    
    # --- Static Batching ---
    static_total_steps = 0
    for i in range(0, n, max_batch_size):
        batch = requests[i:i + max_batch_size]
        static_total_steps += max(batch)
    
    static_total_time = static_total_steps * time_per_step
    static_throughput = n / (static_total_time / 1000.0)
    static_gpu_util = total_tokens / (static_total_steps * max_batch_size)
    
    # --- Continuous Batching ---
    queue = list(requests)
    active = []
    continuous_total_steps = 0
    
    # Fill initial batch
    while len(active) < max_batch_size and queue:
        active.append(queue.pop(0))
    
    while active:
        # Each active sequence generates one token
        continuous_total_steps += 1
        active = [s - 1 for s in active]
        
        # Remove completed sequences
        active = [s for s in active if s > 0]
        
        # Fill empty slots from queue
        while len(active) < max_batch_size and queue:
            active.append(queue.pop(0))
    
    continuous_total_time = continuous_total_steps * time_per_step
    continuous_throughput = n / (continuous_total_time / 1000.0)
    continuous_gpu_util = total_tokens / (continuous_total_steps * max_batch_size)
    
    # Speedup
    speedup = continuous_throughput / static_throughput
    
    return {
        'static_total_time': round(static_total_time, 4),
        'continuous_total_time': round(continuous_total_time, 4),
        'static_throughput': round(static_throughput, 4),
        'continuous_throughput': round(continuous_throughput, 4),
        'static_gpu_utilization': round(static_gpu_util, 4),
        'continuous_gpu_utilization': round(continuous_gpu_util, 4),
        'speedup': round(speedup, 4)
    }