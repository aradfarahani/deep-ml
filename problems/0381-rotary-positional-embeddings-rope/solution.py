import numpy as np

def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    seq_len, d = x.shape
    
    # Compute frequencies for each dimension pair
    i = np.arange(0, d // 2)
    theta = 1.0 / (base ** (2.0 * i / d))
    
    # Compute rotation angles: outer product of positions and frequencies
    angles = positions[:, np.newaxis] * theta[np.newaxis, :]
    
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    # Split into even and odd dimension pairs
    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    
    # Apply 2D rotation to each pair
    x_rot_even = x_even * cos_angles - x_odd * sin_angles
    x_rot_odd = x_even * sin_angles + x_odd * cos_angles
    
    # Interleave even and odd dimensions back together
    result = np.zeros_like(x)
    result[:, 0::2] = x_rot_even
    result[:, 1::2] = x_rot_odd
    
    return result