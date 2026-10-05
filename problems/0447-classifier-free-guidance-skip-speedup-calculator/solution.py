import numpy as np

def cfg_skip_speedup(
    T: int,
    skip_mask: list[bool],
    time_per_pass: float,
    cond_preds: list[float],
    uncond_preds: list[float],
    guidance_scale: float
) -> dict:
    """
    Analyze the speedup and quality impact of skipping unconditional
    passes in Classifier-Free Guidance diffusion inference.
    
    Args:
        T: Total number of denoising timesteps
        skip_mask: Boolean list; True means skip the unconditional pass at that step
        time_per_pass: Time in milliseconds for a single forward pass
        cond_preds: Conditional model predictions at each timestep
        uncond_preds: Unconditional model predictions at each timestep
        guidance_scale: CFG guidance scale (w)
    
    Returns:
        Dictionary with speedup metrics and guided outputs
    """
    total_passes_standard = 2 * T
    total_passes_skipped = 0
    cached_uncond = 0.0
    
    guided_standard = []
    guided_skipped = []
    
    for t in range(T):
        # Standard CFG: always use actual unconditional prediction
        g_std = uncond_preds[t] + guidance_scale * (cond_preds[t] - uncond_preds[t])
        guided_standard.append(g_std)
        
        # Skip strategy
        if skip_mask[t]:
            # Skip unconditional pass, reuse cached value
            effective_uncond = cached_uncond
            total_passes_skipped += 1  # only conditional pass
        else:
            # Perform both passes and update cache
            effective_uncond = uncond_preds[t]
            cached_uncond = uncond_preds[t]
            total_passes_skipped += 2  # both passes
        
        g_skip = effective_uncond + guidance_scale * (cond_preds[t] - effective_uncond)
        guided_skipped.append(g_skip)
    
    speedup_ratio = total_passes_standard / total_passes_skipped if total_passes_skipped > 0 else float('inf')
    time_standard = total_passes_standard * time_per_pass
    time_skipped = total_passes_skipped * time_per_pass
    time_saved = time_standard - time_skipped
    
    max_deviation = max(abs(a - b) for a, b in zip(guided_standard, guided_skipped))
    
    return {
        'total_passes_standard': total_passes_standard,
        'total_passes_skipped': total_passes_skipped,
        'speedup_ratio': round(speedup_ratio, 4),
        'time_saved_ms': round(time_saved, 4),
        'guided_outputs_standard': [round(x, 4) for x in guided_standard],
        'guided_outputs_skipped': [round(x, 4) for x in guided_skipped],
        'max_output_deviation': round(max_deviation, 4)
    }