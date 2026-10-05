def gpu_ops_byte_ratio(gpu_specs: dict) -> dict:
    """
    Compute the ops:byte ratio (ridge point) for each precision
    format from GPU hardware specifications.

    Args:
        gpu_specs: Dictionary with keys:
            - 'compute_tflops': dict mapping precision name -> peak TFLOPS
            - 'memory_bandwidth_gbps': float, peak memory bandwidth in GB/s

    Returns:
        Dictionary mapping each precision name to its ops:byte ratio
        (FLOPs per byte), rounded to 2 decimal places.
    """
    compute = gpu_specs["compute_tflops"]
    bandwidth_gbps = gpu_specs["memory_bandwidth_gbps"]

    ratios = {}
    for precision, tflops in compute.items():
        # TFLOPS = 1e12 ops/s, GB/s = 1e9 bytes/s
        # ratio = (tflops * 1e12) / (bandwidth_gbps * 1e9) = tflops * 1e3 / bandwidth_gbps
        ratio = (tflops * 1e3) / bandwidth_gbps
        ratios[precision] = round(ratio, 2)

    return ratios