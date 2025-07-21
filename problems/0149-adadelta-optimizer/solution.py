import numpy as np

def adadelta_optimizer(parameter, grad, u, v, rho=0.95, epsilon=1e-6):
    """
    Update parameters using the AdaDelta optimizer.
    AdaDelta is an extension of AdaGrad that seeks to reduce its aggressive,
    monotonically decreasing learning rate.
    Args:
        parameter: Current parameter value
        grad: Current gradient
        u: Running average of squared gradients
        v: Running average of squared parameter updates
        rho: Decay rate for the moving average (default=0.95)
        epsilon: Small constant for numerical stability (default=1e-6)
    Returns:
        tuple: (updated_parameter, updated_u, updated_v)
    """
    assert 0 <= rho < 1, "Rho must be between 0 and 1"
    assert epsilon > 0, "Epsilon must be positive"
    assert all(u >= 0) if isinstance(u, np.ndarray) else u >= 0, "u must be non-negative"
    assert all(v >= 0) if isinstance(v, np.ndarray) else v >= 0, "v must be non-negative"

    # Update running average of squared gr