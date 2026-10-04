import numpy as np

def diffusion_process(x_0: np.ndarray, 
                      betas: np.ndarray,
                      timestep: int,
                      forward_noise: np.ndarray,
                      predicted_noise: np.ndarray,
                      backward_noise: np.ndarray = None) -> tuple:
    """
    Implement forward and backward diffusion processes.
    
    Args:
        x_0: Original clean data (any shape)
        betas: Noise schedule array of shape (T,)
        timestep: Current timestep t (1-indexed)
        forward_noise: Noise epsilon for forward diffusion
        predicted_noise: Model's predicted noise for backward diffusion
        backward_noise: Random noise z for stochastic backward step
    
    Returns:
        tuple: (x_t, x_t_minus_1) - noisy and denoised samples
    """
    # Compute alpha values: alpha_t = 1 - beta_t
    alphas = 1.0 - betas
    
    # Compute cumulative products: alpha_bar_t = prod(alpha_s, s=1 to t)
    alpha_bars = np.cumprod(alphas)
    
    # Get values at current timestep (convert to 0-indexed)
    t_idx = timestep - 1
    alpha_bar_t = alpha_bars[t_idx]
    alpha_t = alphas[t_idx]
    beta_t = betas[t_idx]
    
    # Forward process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = np.sqrt(1.0 - alpha_bar_t)
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * forward_noise
    
    # Backward process: compute posterior mean
    # mu = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * predicted_noise)
    coef1 = 1.0 / np.sqrt(alpha_t)
    coef2 = beta_t / sqrt_one_minus_alpha_bar_t
    mu = coef1 * (x_t - coef2 * predicted_noise)
    
    if timestep == 1:
        # At t=1, return mean directly (no noise at final step)
        x_t_minus_1 = mu
    else:
        # Compute posterior variance: sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        alpha_bar_t_minus_1 = alpha_bars[t_idx - 1]
        sigma_t_sq = beta_t * (1.0 - alpha_bar_t_minus_1) / (1.0 - alpha_bar_t)
        sigma_t = np.sqrt(sigma_t_sq)
        
        # x_{t-1} = mu + sigma_t * z
        x_t_minus_1 = mu + sigma_t * backward_noise
    
    return x_t, x_t_minus_1