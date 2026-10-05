import numpy as np

def denoising_score_matching_loss(X: np.ndarray, noisy_X: np.ndarray, sigma: float, predicted_scores: np.ndarray) -> dict:
    """
    Compute the denoising score matching loss for score-based diffusion models.
    
    Args:
        X: Clean data points, shape (n_samples, d)
        noisy_X: Noisy data points, shape (n_samples, d)
        sigma: Noise standard deviation (positive scalar)
        predicted_scores: Model score predictions at noisy points, shape (n_samples, d)
    
    Returns:
        Dictionary with keys 'target_scores', 'loss', 'weighted_loss'
    """
    # Compute the target score: gradient of log p_sigma(noisy_x | x)
    # For Gaussian perturbation kernel N(noisy_x; x, sigma^2 I):
    # nabla_{noisy_x} log p(noisy_x | x) = -(noisy_x - x) / sigma^2
    target_scores = -(noisy_X - X) / (sigma ** 2)
    
    # Compute per-sample squared L2 norm of score difference
    diff = predicted_scores - target_scores
    per_sample_loss = np.sum(diff ** 2, axis=-1)  # shape (n_samples,)
    
    # Average over samples
    loss = float(np.mean(per_sample_loss))
    
    # Weighted loss for multi-scale training (sigma^2 weighting)
    weighted_loss = float((sigma ** 2) * loss)
    
    return {
        'target_scores': target_scores,
        'loss': loss,
        'weighted_loss': weighted_loss
    }