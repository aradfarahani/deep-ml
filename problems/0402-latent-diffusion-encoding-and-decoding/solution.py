import numpy as np

def latent_encode_decode(x, W_enc_mu, b_enc_mu, W_enc_logvar, b_enc_logvar, W_dec, b_dec, epsilon):
    """
    Perform variational encoding, latent sampling, and decoding
    as used in latent diffusion models.
    """
    # Encode: project input to mean and log-variance in latent space
    mu = x @ W_enc_mu + b_enc_mu
    log_var = x @ W_enc_logvar + b_enc_logvar
    
    # Reparameterization trick: sample z from N(mu, sigma^2)
    std = np.exp(0.5 * log_var)
    z = mu + std * epsilon
    
    # Decode: map latent sample back to input space
    x_recon = z @ W_dec + b_dec
    
    return {
        'mu': mu,
        'log_var': log_var,
        'z': z,
        'x_recon': x_recon
    }