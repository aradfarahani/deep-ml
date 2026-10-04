import numpy as np

def diffusion_loss(x_0: np.ndarray, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: np.ndarray, predicted_noise: np.ndarray) -> float:
    """
    Compute the reconstruction loss for diffusion model training.
    """
    # Create linear beta schedule
    betas = np.linspace(beta_start, beta_end, num_timesteps)
    
    # Compute alphas = 1 - betas
    alphas = 1.0 - betas
    
    # Compute cumulative product of alphas (alpha_bar)
    alpha_bar = np.cumprod(alphas)
    
    # Get alpha_bar at timestep t (convert to 0-indexed)
    alpha_bar_t = alpha_bar[t - 1]
    
    # Compute coefficients
    sqrt_alpha_bar = np.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = np.sqrt(1.0 - alpha_bar_t)
    
    # Compute x_t using forward diffusion
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    # Reconstruct x_0 from x_t using predicted noise
    x_0_reconstructed = (x_t - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar
    
    # Compute MSE loss
    loss = np.mean((x_0 - x_0_reconstructed) ** 2)
    
    return round(loss, 4)