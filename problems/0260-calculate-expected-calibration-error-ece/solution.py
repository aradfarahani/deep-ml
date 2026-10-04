import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
        y_true: List or array of true binary labels (0 or 1)
        y_prob: List or array of predicted probabilities for the positive class
        n_bins: Number of bins for grouping predictions (default: 10)
    
    Returns:
        float: ECE value rounded to 3 decimal places
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    n_samples = len(y_true)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        # First bin includes both boundaries, others exclude lower
        if i == 0:
            in_bin = (y_prob >= lower) & (y_prob <= upper)
        else:
            in_bin = (y_prob > lower) & (y_prob <= upper)
        
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            # Average accuracy in bin (fraction of positive labels)
            bin_accuracy = np.mean(y_true[in_bin])
            # Average confidence in bin (mean predicted probability)
            bin_confidence = np.mean(y_prob[in_bin])
            # Weighted absolute difference
            ece += (bin_size / n_samples) * np.abs(bin_accuracy - bin_confidence)
    
    return round(ece, 3)