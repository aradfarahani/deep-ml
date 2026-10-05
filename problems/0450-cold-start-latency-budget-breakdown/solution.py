def cold_start_breakdown(config: dict) -> dict:
    """
    Break down the cold start latency of an ML model serving instance.

    Args:
        config: Dictionary with deployment parameters.

    Returns:
        Dictionary with per-phase latencies, total, SLA status,
        budget usage percentage, and bottleneck phase name.
    """
    model_size_gb = float(config['model_size_gb'])
    storage_bandwidth_gbps = float(config['storage_bandwidth_gbps'])
    runtime_init_ms = float(config['runtime_init_ms'])
    compilation_ms_per_gb = float(config['compilation_ms_per_gb'])
    warmup_batches = int(config['warmup_batches'])
    warmup_batch_ms = float(config['warmup_batch_ms'])
    health_check_ms = float(config['health_check_ms'])
    sla_budget_ms = float(config['sla_budget_ms'])

    weight_load_ms = round((model_size_gb / storage_bandwidth_gbps) * 1000.0, 2)
    compilation_ms = round(model_size_gb * compilation_ms_per_gb, 2)
    warmup_ms = round(warmup_batches * warmup_batch_ms, 2)
    runtime_init_ms = round(runtime_init_ms, 2)
    health_check_ms = round(health_check_ms, 2)

    phases = {
        'runtime_init_ms': runtime_init_ms,
        'weight_load_ms': weight_load_ms,
        'compilation_ms': compilation_ms,
        'warmup_ms': warmup_ms,
        'health_check_ms': health_check_ms
    }

    total_ms = round(sum(phases.values()), 2)
    meets_sla = total_ms <= sla_budget_ms
    budget_used_pct = round((total_ms / sla_budget_ms) * 100.0, 2)
    bottleneck = max(phases, key=phases.get)

    result = dict(phases)
    result['total_ms'] = total_ms
    result['meets_sla'] = meets_sla
    result['budget_used_pct'] = budget_used_pct
    result['bottleneck'] = bottleneck
    return result