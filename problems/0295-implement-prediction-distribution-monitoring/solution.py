import numpy as np

def monitor_prediction_distribution(reference_preds: list, current_preds: list, n_bins: int = 10) -> dict:
    """
    Monitor prediction distribution changes between reference and current predictions.
    
    Args:
        reference_preds: List of reference prediction scores (floats between 0 and 1)
        current_preds: List of current prediction scores (floats between 0 and 1)
        n_bins: Number of bins for histogram comparison
    
    Returns:
        Dictionary with keys: 'mean_shift', 'std_ratio', 'js_divergence', 'drift_detected'
    """
    ref = np.array(reference_preds, dtype=float)
    curr = np.array(current_preds, dtype=float)
    
    # Mean shift
    mean_shift = float(np.mean(curr) - np.mean(ref))
    
    # Standard deviation ratio
    ref_std = float(np.std(ref))
    curr_std = float(np.std(curr))
    std_ratio = curr_std / ref_std if ref_std > 0 else float('inf')
    
    # Create histograms
    bins = np.linspace(0, 1, n_bins + 1)
    ref_counts, _ = np.histogram(ref, bins=bins)
    curr_counts, _ = np.histogram(curr, bins=bins)
    
    # Convert to probabilities with Laplace smoothing
    ref_prob = (ref_counts + 1) / (len(ref) + n_bins)
    curr_prob = (curr_counts + 1) / (len(curr) + n_bins)
    
    # Jensen-Shannon divergence
    m = 0.5 * (ref_prob + curr_prob)
    js_div = 0.5 * np.sum(ref_prob * np.log(ref_prob / m)) + 0.5 * np.sum(curr_prob * np.log(curr_prob / m))
    
    # Drift threshold
    drift_detected = bool(js_div > 0.1)
    
    return {
        'mean_shift': round(mean_shift, 4),
        'std_ratio': round(std_ratio, 4),
        'js_divergence': round(float(js_div), 4),
        'drift_detected': drift_detected
    }