import numpy as np

def unipc_step(
    x_t: np.ndarray,
    eps_t: np.ndarray,
    eps_prev: np.ndarray,
    alpha_t: float,
    sigma_t: float,
    alpha_prev: float,
    sigma_prev: float
) -> tuple:
    # Estimate clean sample from current noisy input
    x0_pred = (x_t - sigma_t * eps_t) / alpha_t

    # Predictor: standard DDIM step using eps_t
    x_predictor = alpha_prev * x0_pred + sigma_prev * eps_t

    # Corrector: blend current and previous eps, reproject
    eps_blended = (eps_t + eps_prev) / 2.0
    x_corrector = alpha_prev * x0_pred + sigma_prev * eps_blended

    return x_predictor, x_corrector