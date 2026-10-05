def classify_llm_phases(num_params: int, sequence_length: int, batch_size: int, bytes_per_param: int, peak_flops: float, peak_bandwidth: float) -> dict:
    ridge_point = peak_flops / peak_bandwidth

    def analyze_phase(tokens):
        total_flops = 2 * num_params * tokens
        memory_bytes = num_params * bytes_per_param
        ai = total_flops / memory_bytes
        if ai >= ridge_point:
            bottleneck = 'compute-bound'
            achieved = float(peak_flops)
        else:
            bottleneck = 'memory-bound'
            achieved = ai * peak_bandwidth
        utilization = (achieved / peak_flops) * 100
        return {
            'total_flops': total_flops,
            'memory_bytes': memory_bytes,
            'arithmetic_intensity': round(ai, 4),
            'bottleneck': bottleneck,
            'achieved_flops': round(achieved, 4),
            'utilization_percent': round(utilization, 4)
        }

    return {
        'ridge_point': round(ridge_point, 4),
        'prefill': analyze_phase(sequence_length),
        'decode': analyze_phase(batch_size)
    }