def mig_resource_allocation(gpu_config: dict, mig_profiles: list, workloads: list) -> dict:
    """
    Allocate MIG instances to workloads on a single GPU.
    
    Args:
        gpu_config: Dict with 'total_compute_slices' and 'total_memory_gb'
        mig_profiles: List of available MIG profile dicts
        workloads: List of workload requirement dicts
    
    Returns:
        Dict with allocation results and utilization metrics
    """
    total_compute = gpu_config['total_compute_slices']
    total_memory = gpu_config['total_memory_gb']
    
    remaining_compute = total_compute
    remaining_memory = total_memory
    
    sorted_workloads = sorted(workloads, key=lambda w: (-w['min_compute_slices'], -w['min_memory_gb']))
    
    allocations = []
    rejected = []
    
    for wl in sorted_workloads:
        suitable = [p for p in mig_profiles
                    if p['compute_slices'] >= wl['min_compute_slices']
                    and p['memory_gb'] >= wl['min_memory_gb']]
        
        suitable.sort(key=lambda p: (p['compute_slices'], p['memory_gb']))
        
        allocated = False
        for profile in suitable:
            if (profile['compute_slices'] <= remaining_compute
                and profile['memory_gb'] <= remaining_memory):
                allocations.append({
                    'workload': wl['name'],
                    'profile': profile['name'],
                    'compute_slices': profile['compute_slices'],
                    'memory_gb': profile['memory_gb']
                })
                remaining_compute -= profile['compute_slices']
                remaining_memory -= profile['memory_gb']
                allocated = True
                break
        
        if not allocated:
            rejected.append(wl['name'])
    
    total_compute_used = total_compute - remaining_compute
    total_memory_used = total_memory - remaining_memory
    
    compute_util = (total_compute_used / total_compute) * 100 if total_compute > 0 else 0.0
    memory_util = (total_memory_used / total_memory) * 100 if total_memory > 0 else 0.0
    
    return {
        'allocations': allocations,
        'total_compute_used': total_compute_used,
        'total_memory_used': round(total_memory_used, 2),
        'compute_utilization': round(compute_util, 2),
        'memory_utilization': round(memory_util, 2),
        'workloads_served': len(allocations),
        'workloads_rejected': rejected
    }