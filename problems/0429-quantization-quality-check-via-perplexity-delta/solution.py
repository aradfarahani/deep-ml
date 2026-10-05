import numpy as np

def quantization_quality_check(log_probs_original: list, log_probs_quantized: list, thresholds: dict = None) -> dict:
    """
    Evaluate quantization quality by comparing perplexity before and after quantization.
    
    Args:
        log_probs_original: Token-level log-probabilities from the original (full precision) model
        log_probs_quantized: Token-level log-probabilities from the quantized model
        thresholds: Optional dict with 'excellent' and 'acceptable' percentage thresholds
    
    Returns:
        dict with perplexity metrics and quality classification
    """
    if thresholds is None:
        thresholds = {'excellent': 1.0, 'acceptable': 5.0}
    
    log_probs_orig = np.array(log_probs_original, dtype=np.float64)
    log_probs_quant = np.array(log_probs_quantized, dtype=np.float64)
    
    N = len(log_probs_orig)
    
    pp_original = float(np.exp(-np.sum(log_probs_orig) / N))
    pp_quantized = float(np.exp(-np.sum(log_probs_quant) / N))
    
    delta = pp_quantized - pp_original
    relative_delta = (delta / pp_original) * 100.0
    
    if relative_delta <= thresholds['excellent']:
        quality = 'excellent'
    elif relative_delta <= thresholds['acceptable']:
        quality = 'acceptable'
    else:
        quality = 'poor'
    
    return {
        'perplexity_original': round(pp_original, 4),
        'perplexity_quantized': round(pp_quantized, 4),
        'perplexity_delta': round(delta, 4),
        'relative_delta_percent': round(relative_delta, 4),
        'quality': quality
    }