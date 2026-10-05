def expert_parallel_comm_cost(num_devices: int, num_experts: int, tokens_per_device: list, token_size_bytes: int, bandwidth_gbps: float = None) -> dict:
    experts_per_device = num_experts // num_devices
    
    # Build dispatch matrix: dispatch[i][j] = tokens device i sends to device j
    dispatch_matrix = [[0] * num_devices for _ in range(num_devices)]
    
    for src_device, expert_list in enumerate(tokens_per_device):
        for expert_id in expert_list:
            dst_device = expert_id // experts_per_device
            dispatch_matrix[src_device][dst_device] += 1
    
    # Total communication bytes (non-local only)
    total_comm_tokens = 0
    for i in range(num_devices):
        for j in range(num_devices):
            if i != j:
                total_comm_tokens += dispatch_matrix[i][j]
    total_comm_bytes = total_comm_tokens * token_size_bytes
    
    # Per-device send and receive bytes (non-local)
    send_per_device = []
    recv_per_device = []
    for i in range(num_devices):
        send = sum(dispatch_matrix[i][j] for j in range(num_devices) if j != i)
        recv = sum(dispatch_matrix[j][i] for j in range(num_devices) if j != i)
        send_per_device.append(send * token_size_bytes)
        recv_per_device.append(recv * token_size_bytes)
    
    max_send_bytes = max(send_per_device)
    max_recv_bytes = max(recv_per_device)
    
    # Load per device: total tokens processed by experts on each device
    load_per_device = [0] * num_devices
    for dst in range(num_devices):
        for src in range(num_devices):
            load_per_device[dst] += dispatch_matrix[src][dst]
    
    total_tokens = sum(load_per_device)
    avg_load = total_tokens / num_devices if num_devices > 0 else 0
    max_load = max(load_per_device)
    load_imbalance = round(max_load / avg_load, 4) if avg_load > 0 else 0.0
    
    result = {
        'dispatch_matrix': dispatch_matrix,
        'total_comm_bytes': total_comm_bytes,
        'max_device_send_bytes': max_send_bytes,
        'max_device_recv_bytes': max_recv_bytes,
        'load_per_device': load_per_device,
        'load_imbalance': load_imbalance
    }
    
    if bandwidth_gbps is not None:
        bottleneck_bytes = max(max_send_bytes, max_recv_bytes)
        comm_time_ms = round((bottleneck_bytes / (bandwidth_gbps * 1e9)) * 1000, 4)
        result['comm_time_ms'] = comm_time_ms
    
    return result

    