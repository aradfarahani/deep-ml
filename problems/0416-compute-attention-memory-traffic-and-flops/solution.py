def attention_memory_flops(B: int, h: int, N: int, d: int, bytes_per_element: int = 2) -> dict:
    """
    Compute memory traffic and FLOPs for standard self-attention.

    Args:
        B: Batch size
        h: Number of attention heads
        N: Sequence length
        d: Head dimension
        bytes_per_element: Bytes per element (e.g., 2 for FP16, 4 for FP32)

    Returns:
        dict with keys:
            'qk_flops': int - FLOPs for Q @ K^T
            'softmax_flops': int - FLOPs for softmax
            'pv_flops': int - FLOPs for P @ V
            'total_flops': int - Total FLOPs
            'memory_bytes': int - Total memory traffic in bytes
            'arithmetic_intensity': float - FLOPs per byte, rounded to 2 decimal places
    """
    # FLOPs computation
    # Q @ K^T: (N x d) @ (d x N) = 2*N*d*N per head per batch
    qk_flops = 2 * B * h * N * N * d
    
    # Softmax: 5 ops per element per row (max, subtract, exp, sum, divide)
    # N rows each of length N, per head per batch
    softmax_flops = 5 * B * h * N * N
    
    # P @ V: (N x N) @ (N x d) = 2*N*N*d per head per batch
    pv_flops = 2 * B * h * N * N * d
    
    total_flops = qk_flops + softmax_flops + pv_flops
    
    # Memory traffic (in elements)
    # Step 1 (Q @ K^T): Read Q (BhNd) + Read K (BhNd) + Write S (BhNN)
    step1_elements = B * h * (2 * N * d + N * N)
    
    # Step 2 (softmax): Read S (BhNN) + Write P (BhNN)
    step2_elements = B * h * (2 * N * N)
    
    # Step 3 (P @ V): Read P (BhNN) + Read V (BhNd) + Write O (BhNd)
    step3_elements = B * h * (N * N + 2 * N * d)
    
    total_elements = step1_elements + step2_elements + step3_elements
    # = B * h * (4*N*d + 4*N*N) = 4*B*h*N*(d + N)
    memory_bytes = total_elements * bytes_per_element
    
    arithmetic_intensity = round(total_flops / memory_bytes, 2)
    
    return {
        'qk_flops': qk_flops,
        'softmax_flops': softmax_flops,
        'pv_flops': pv_flops,
        'total_flops': total_flops,
        'memory_bytes': memory_bytes,
        'arithmetic_intensity': arithmetic_intensity
    }