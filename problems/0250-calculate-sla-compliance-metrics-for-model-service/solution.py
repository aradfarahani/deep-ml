def calculate_sla_metrics(requests: list, latency_sla_ms: float = 100.0) -> dict:
    """
    Calculate SLA compliance metrics for a model serving endpoint.
    
    Args:
        requests: list of request results, each a dict with 'latency_ms' and 'status'
        latency_sla_ms: maximum acceptable latency in ms for SLA compliance
    
    Returns:
        dict with keys: 'latency_sla_compliance', 'error_rate', 'overall_sla_compliance'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    if not requests:
        return {}
    
    total = len(requests)
    successful = [r for r in requests if r.get('status') == 'success']
    success_count = len(successful)
    
    # Count errors and timeouts
    error_count = sum(1 for r in requests if r.get('status') in ['error', 'timeout'])
    
    # Successful requests that met latency SLA
    met_sla = [r for r in successful if r.get('latency_ms', float('inf')) <= latency_sla_ms]
    met_sla_count = len(met_sla)
    
    # Latency SLA compliance (among successful requests)
    if success_count > 0:
        latency_sla_compliance = (met_sla_count / success_count) * 100
    else:
        latency_sla_compliance = 0.0
    
    # Error rate
    error_rate = (error_count / total) * 100
    
    # Overall SLA compliance (succeeded AND met latency)
    overall_sla_compliance = (met_sla_count / total) * 100
    
    return {
        'latency_sla_compliance': round(latency_sla_compliance, 2),
        'error_rate': round(error_rate, 2),
        'overall_sla_compliance': round(overall_sla_compliance, 2)
    }
    