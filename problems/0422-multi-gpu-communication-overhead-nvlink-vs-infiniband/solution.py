import math

def multi_gpu_comm_overhead(message_size_gb: float, num_gpus: int,
                           nvlink_config: dict, ib_config: dict,
                           operation: str) -> dict:
    """
    Compare communication overhead between NVLink and InfiniBand
    for collective GPU operations.
    
    Args:
        message_size_gb: Data payload size in GB
        num_gpus: Number of GPUs
        nvlink_config: Dict with 'bandwidth_gbps' and 'latency_us'
        ib_config: Dict with 'bandwidth_gbps' and 'latency_us'
        operation: One of 'all_reduce', 'all_gather', 'broadcast'
    
    Returns:
        Dict with timing comparison and bottleneck analysis
    """
    def compute_time(msg_gb, n_gpus, bw_gbps, lat_us, op):
        if op == 'all_reduce':
            num_steps = 2 * (n_gpus - 1)
            data_per_step = msg_gb / n_gpus
        elif op == 'all_gather':
            num_steps = n_gpus - 1
            data_per_step = msg_gb / n_gpus
        elif op == 'broadcast':
            num_steps = math.ceil(math.log2(n_gpus))
            data_per_step = msg_gb
        else:
            raise ValueError(f"Unknown operation: {op}")
        
        latency_ms = num_steps * lat_us * 0.001
        bandwidth_ms = num_steps * data_per_step / bw_gbps * 1000
        total_ms = latency_ms + bandwidth_ms
        bottleneck = 'bandwidth' if bandwidth_ms >= latency_ms else 'latency'
        
        return total_ms, bottleneck
    
    nv_total, nv_btn = compute_time(message_size_gb, num_gpus,
        nvlink_config['bandwidth_gbps'], nvlink_config['latency_us'], operation)
    
    ib_total, ib_btn = compute_time(message_size_gb, num_gpus,
        ib_config['bandwidth_gbps'], ib_config['latency_us'], operation)
    
    speedup = ib_total / nv_total
    
    return {
        'nvlink_time_ms': round(nv_total, 4),
        'ib_time_ms': round(ib_total, 4),
        'speedup': round(speedup, 4),
        'nvlink_bottleneck': nv_btn,
        'ib_bottleneck': ib_btn
    }