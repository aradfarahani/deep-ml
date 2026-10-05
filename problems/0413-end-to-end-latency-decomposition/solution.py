import numpy as np

def decompose_latency(stage_latencies: dict, percentiles: list) -> dict:
    """
    Decompose end-to-end inference latency into component stages.
    
    Args:
        stage_latencies: dict mapping stage name -> np.ndarray of latency measurements (ms)
        percentiles: list of percentile values to compute (e.g., [50, 95, 99])
    
    Returns:
        Dictionary with keys: 'e2e_mean', 'e2e_percentiles', 'stage_stats',
                               'bottleneck', 'stage_pct'
    """
    stages = list(stage_latencies.keys())
    n_requests = len(stage_latencies[stages[0]])
    
    # Compute per-request end-to-end latency
    e2e = np.zeros(n_requests)
    for stage in stages:
        e2e = e2e + np.array(stage_latencies[stage], dtype=float)
    
    e2e_mean = round(float(np.mean(e2e)), 2)
    
    # E2E percentiles
    e2e_pcts = {}
    for p in percentiles:
        e2e_pcts[p] = round(float(np.percentile(e2e, p)), 2)
    
    # Per-stage statistics
    stage_stats = {}
    for stage in stages:
        data = np.array(stage_latencies[stage], dtype=float)
        s = {
            'mean': round(float(np.mean(data)), 2),
            'std': round(float(np.std(data)), 2),
            'percentiles': {}
        }
        for p in percentiles:
            s['percentiles'][p] = round(float(np.percentile(data, p)), 2)
        stage_stats[stage] = s
    
    # Identify bottleneck (stage with highest mean)
    bottleneck = max(stages, key=lambda s: np.mean(stage_latencies[s]))
    
    # Stage percentage contribution to mean E2E
    total_mean = sum(float(np.mean(stage_latencies[s])) for s in stages)
    stage_pct = {}
    for stage in stages:
        stage_pct[stage] = round(float(np.mean(stage_latencies[stage])) / total_mean * 100, 2)
    
    return {
        'e2e_mean': e2e_mean,
        'e2e_percentiles': e2e_pcts,
        'stage_stats': stage_stats,
        'bottleneck': bottleneck,
        'stage_pct': stage_pct
    }