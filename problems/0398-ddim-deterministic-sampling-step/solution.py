import numpy as np

def ddim_sample_step(x_t: np.ndarray, noise_pred: np.ndarray, alpha_bar_t: float, alpha_bar_t_prev: float) -> tuple:
    """
    Perform one deterministic DDIM sampling step.
    
    Args:
        x_t: Noisy sample at timestep t, shape (D,)
        noise_pred: Predicted noise from the model, shape (D,)
        alpha_bar_t: Cumulative alpha at timestep t
        alpha_bar_t_prev: Cumulative alpha at timestep t-1
    
    Returns:
        Tuple of (pred_x0, x_t_prev), each a numpy array rounded to 4 decimals
    """
    # Step 1: Predict the original clean sample x_0
    pred_x0 = (x_t - np.sqrt(1.0 - alpha_bar_t) * noise_pred) / np.sqrt(alpha_bar_t)
    
    # Step 2: Compute the deterministic x_{t-1} using the DDIM update rule
    # With eta=0 (deterministic), this simplifies to:
    # x_{t-1} = sqrt(alpha_bar_{t-1}) * pred_x0 + sqrt(1 - alpha_bar_{t-1}) * noise_pred
    x_t_prev = np.sqrt(alpha_bar_t_prev) * pred_x0 + np.sqrt(1.0 - alpha_bar_t_prev) * noise_pred
    
    return (np.round(pred_x0, 4), np.round(x_t_prev, 4))