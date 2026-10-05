import numpy as np

def ddpm_reverse_step(x_t: np.ndarray, predicted_noise: np.ndarray, t: int,
                     betas: np.ndarray, noise: np.ndarray = None) -> np.ndarray:
    """
    Perform a single reverse (denoising) step of the DDPM sampling process.
    
    Args:
        x_t: Noisy sample at timestep t, shape (D,)
        predicted_noise: Model's noise prediction, shape (D,)
        t: Current timestep (1-indexed)
        betas: Noise schedule array, shape (T,)
        noise: Optional noise array for stochastic component, shape (D,)
    
    Returns:
        Denoised sample x_{t-1}, shape (D,)
    """
    beta_t = betas[t - 1]
    alpha_t = 1.0 - beta_t
    
    # Cumulative product of alphas from step 1 to t
    alpha_bar_t = np.prod(1.0 - betas[:t])
    
    # Coefficient for predicted noise in the mean formula
    noise_coeff = beta_t / np.sqrt(1.0 - alpha_bar_t)
    
    # Compute the posterior mean
    mean = (1.0 / np.sqrt(alpha_t)) * (x_t - noise_coeff * predicted_noise)
    
    # For t > 1, add stochastic noise; for t = 1, return the mean directly
    if t > 1:
        sigma_t = np.sqrt(beta_t)
        if noise is None:
            noise = np.random.randn(*x_t.shape)
        return mean + sigma_t * noise
    else:
        return mean