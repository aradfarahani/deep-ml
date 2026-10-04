import numpy as np

def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """
    Calculate P50, P95, and P99 latency percentiles.
    
    Args:
        latencies: List of latency measurements
    
    Returns:
        Dictionary with keys 'P50', 'P95', 'P99' containing
        the respective percentile values rounded to 4 decimal places
    """
    if not latencies:
        return {"P50": 0.0, "P95": 0.0, "P99": 0.0}
    
    latencies_arr = np.array(latencies)
    
    p50 = float(np.percentile(latencies_arr, 50))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))
    
    return {
        "P50": round(p50, 4),
        "P95": round(p95, 4),
        "P99": round(p99, 4)
    }