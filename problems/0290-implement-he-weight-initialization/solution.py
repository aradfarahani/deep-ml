import numpy as np

def he_initialization(n_in: int, n_out: int, mode: str = 'fan_in', distribution: str = 'normal', seed: int = None) -> np.ndarray:
    """
    Implement He (Kaiming) weight initialization.
    
    Parameters:
    n_in: number of input units
    n_out: number of output units
    mode: 'fan_in' or 'fan_out'
    distribution: 'normal' or 'uniform'
    seed: random seed for reproducibility
    
    Returns:
    numpy array of shape (n_in, n_out) with He-initialized weights
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Select fan based on mode
    fan = n_in if mode == 'fan_in' else n_out
    
    if distribution == 'normal':
        # He normal: sample from N(0, sqrt(2/fan))
        std = np.sqrt(2.0 / fan)
        weights = np.random.randn(n_in, n_out) * std
    else:
        # He uniform: sample from U(-sqrt(6/fan), sqrt(6/fan))
        limit = np.sqrt(6.0 / fan)
        weights = np.random.uniform(-limit, limit, size=(n_in, n_out))
    
    return weights