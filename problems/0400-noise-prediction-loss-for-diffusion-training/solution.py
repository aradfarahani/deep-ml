import numpy as np

def noise_prediction_loss(x_0, alpha_bar, t, epsilon, epsilon_pred):
    """
    Compute the noisy samples and noise prediction MSE loss for diffusion model training.
    
    Args:
        x_0: Clean data samples, shape (B, D)
        alpha_bar: Cumulative noise schedule, shape (T,)
        t: Timestep indices for each sample, shape (B,)
        epsilon: True Gaussian noise, shape (B, D)
        epsilon_pred: Predicted noise from model, shape (B, D)
    
    Returns:
        tuple: (x_t, loss) where x_t has shape (B, D) and loss is a scalar float
    """
    # Get alpha_bar values for each sample's timestep
    alpha_bar_t = alpha_bar[t]  # shape (B,)
    
    # Reshape for broadcasting: (B,) -> (B, 1)
    alpha_bar_t = alpha_bar_t.reshape(-1, 1)
    
    # Construct noisy samples using the reparameterization trick
    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1.0 - alpha_bar_t) * epsilon
    
    # Compute MSE loss between true noise and predicted noise
    loss = float(np.mean((epsilon - epsilon_pred) ** 2))
    
    return x_t, loss