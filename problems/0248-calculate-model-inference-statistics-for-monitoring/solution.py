def calculate_inference_stats(latencies_ms: list) -> dict:
    """
    Calculate inference statistics for model monitoring.
    
    Args:
        latencies_ms: list of latency measurements in milliseconds
    
    Returns:
        dict with keys: 'throughput_per_sec', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms'
        All values rounded to 2 decimal places.
    """
    if len(latencies_ms) == 0:
        return {}
    
    n = len(latencies_ms)
    sorted_latencies = sorted(latencies_ms)
    
    # Average latency
    avg_latency = sum(latencies_ms) / n
    
    # Throughput (requests per second for sequential processing)
    throughput = 1000 / avg_latency if avg_latency > 0 else 0
    
    # Percentile calculation using linear interpolation
    def percentile(sorted_data, p):
        if len(sorted_data) == 1:
            return float(sorted_data[0])
        idx = (p / 100) * (len(sorted_data) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(sorted_data) - 1)
        fraction = idx - lower
        return sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])
    
    p50 = percentile(sorted_latencies, 50)
    p95 = percentile(sorted_latencies, 95)
    p99 = percentile(sorted_latencies, 99)
    
    return {
        'throughput_per_sec': round(throughput, 2),
        'avg_latency_ms': round(avg_latency, 2),
        'p50_ms': round(p50, 2),
        'p95_ms': round(p95, 2),
        'p99_ms': round(p99, 2)
    }