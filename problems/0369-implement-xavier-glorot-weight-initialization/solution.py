import numpy as np

def xavier_init(fan_in: int, fan_out: int, mode: str = 'uniform', seed: int = 42) -> dict:
    """
    Perform Xavier/Glorot weight initialization.

    Args:
        fan_in (int): Number of input units.
        fan_out (int): Number of output units.
        mode (str): 'uniform' or 'normal'.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Contains 'weights' (nested list), 'shape' (list), and 'param' (float).
    """
    np.random.seed(seed)
    
    if mode == 'uniform':
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        weights = np.random.uniform(-limit, limit, (fan_in, fan_out))
        return {
            'weights': np.round(weights, 4).tolist(),
            'shape': [fan_in, fan_out],
            'param': round(float(limit), 4)
        }
    elif mode == 'normal':
        std = np.sqrt(2.0 / (fan_in + fan_out))
        weights = np.random.normal(0, std, (fan_in, fan_out))
        return {
            'weights': np.round(weights, 4).tolist(),
            'shape': [fan_in, fan_out],
            'param': round(float(std), 4)
        }
    else:
        raise ValueError("mode must be 'uniform' or 'normal'")