def kernel_fusion_savings(input_elements: int, operations: list, dtype_bytes: int = 4, memory_bandwidth_gbps: float = None) -> dict:
    """
    Calculate memory traffic savings from fusing a chain of GPU kernel operations.
    
    Args:
        input_elements: Number of elements in the initial input tensor
        operations: List of dicts with 'output_elements' and optional 'extra_param_elements'
        dtype_bytes: Bytes per element (default: 4 for float32)
        memory_bandwidth_gbps: Optional GPU memory bandwidth in GB/s
    
    Returns:
        Dict with memory traffic analysis and optional timing estimates
    """
    prev_output_elements = input_elements
    unfused_total = 0
    total_extra_params = 0
    
    for op in operations:
        out_elem = op['output_elements']
        extra_params = op.get('extra_param_elements', 0)
        
        # Unfused: each kernel reads its input + params from global memory, writes output
        read_bytes = (prev_output_elements + extra_params) * dtype_bytes
        write_bytes = out_elem * dtype_bytes
        unfused_total += read_bytes + write_bytes
        
        total_extra_params += extra_params
        prev_output_elements = out_elem
    
    # Fused: read initial input + all params once, write only final output
    fused_total = (input_elements + total_extra_params) * dtype_bytes + operations[-1]['output_elements'] * dtype_bytes
    
    saved = unfused_total - fused_total
    savings_pct = round((saved / unfused_total) * 100, 2) if unfused_total > 0 else 0.0
    
    result = {
        'unfused_memory_bytes': unfused_total,
        'fused_memory_bytes': fused_total,
        'memory_saved_bytes': saved,
        'savings_percent': savings_pct
    }
    
    if memory_bandwidth_gbps is not None:
        unfused_time_ms = round((unfused_total / (memory_bandwidth_gbps * 1e9)) * 1000, 4)
        fused_time_ms = round((fused_total / (memory_bandwidth_gbps * 1e9)) * 1000, 4)
        speedup = round(unfused_total / fused_total, 4) if fused_total > 0 else float('inf')
        result['unfused_time_ms'] = unfused_time_ms
        result['fused_time_ms'] = fused_time_ms
        result['speedup'] = speedup
    
    return result
    