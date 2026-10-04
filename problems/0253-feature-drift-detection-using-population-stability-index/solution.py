import numpy as np

def detect_feature_drift(reference_data: list, production_data: list, num_bins: int = 10) -> dict:
    """
    Detect feature drift using Population Stability Index (PSI).
    
    Args:
        reference_data: List of feature values from reference distribution (e.g., training)
        production_data: List of feature values from production distribution
        num_bins: Number of bins for histogram comparison
    
    Returns:
        dict with 'psi', 'drift_detected', and 'drift_level'
    """
    if len(reference_data) == 0 or len(production_data) == 0:
        return {}
    
    reference = np.array(reference_data, dtype=float)
    production = np.array(production_data, dtype=float)
    
    # Determine bin edges from combined data range
    min_val = min(reference.min(), production.min())
    max_val = max(reference.max(), production.max())
    
    # Handle case where all values are identical
    if min_val == max_val:
        return {'psi': 0.0, 'drift_detected': False, 'drift_level': 'none'}
    
    # Create bin edges
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    # Calculate histograms
    ref_hist, _ = np.histogram(reference, bins=bin_edges)
    prod_hist, _ = np.histogram(production, bins=bin_edges)
    
    # Convert to proportions
    ref_pct = ref_hist / len(reference)
    prod_pct = prod_hist / len(production)
    
    # Replace zeros with small epsilon to avoid log(0)
    epsilon = 0.0001
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    prod_pct = np.where(prod_pct == 0, epsilon, prod_pct)
    
    # Calculate PSI: sum of (prod% - ref%) * ln(prod% / ref%)
    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    
    # Determine drift level based on industry thresholds
    if psi < 0.1:
        drift_level = 'none'
        drift_detected = False
    elif psi < 0.25:
        drift_level = 'moderate'
        drift_detected = True
    else:
        drift_level = 'significant'
        drift_detected = True
    
    return {
        'psi': round(float(psi), 4),
        'drift_detected': drift_detected,
        'drift_level': drift_level
    }