import numpy as np

def vae_loss(x: np.ndarray, x_reconstructed: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> tuple:
    """
    Compute the VAE loss (negative ELBO).

    Args:
        x: np.ndarray of shape (batch_size, features), original input
        x_reconstructed: np.ndarray of shape (batch_size, features), reconstructed input
        mu: np.ndarray of shape (batch_size, latent_dim), latent mean
        log_var: np.ndarray of shape (batch_size, latent_dim), latent log-variance

    Returns:
        tuple: (total_loss, reconstruction_loss, kl_divergence) as floats
    """
    # Reconstruction loss: mean over batch of per-sample sum of squared errors
    recon_loss = np.mean(np.sum((x - x_reconstructed) ** 2, axis=1))

    # KL divergence: mean over batch of per-sample KL
    kl_div = -0.5 * np.mean(np.sum(1.0 + log_var - mu ** 2 - np.exp(log_var), axis=1))

    total_loss = recon_loss + kl_div

    # Adding 0.0 converts IEEE 754 -0.0 to 0.0
    return (float(total_loss) + 0.0, float(recon_loss) + 0.0, float(kl_div) + 0.0)