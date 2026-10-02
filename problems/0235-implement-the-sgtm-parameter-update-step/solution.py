import numpy as np


def sgtm_step(
    params: np.ndarray,
    grad: np.ndarray,
    forget_mask: np.ndarray,
    lr: float = 0.1,
    batch_type: str = "forget",
) -> np.ndarray:
    """Perform one SGTM update step on a 1D parameter vector.

    SGTM splits parameters into 'forget' and 'retain' groups using a binary mask.
    Depending on the batch type, only some parameters are allowed to update.

    Args:
        params: Current parameters, shape (d,)
        grad: Gradient for this batch, shape (d,)
        forget_mask: Mask for forget parameters, same shape as params. Any
            nonzero entry is treated as 1 (forget), zero is retain.
        lr: Learning rate for the update step.
        batch_type: One of {'forget', 'retain', 'unlabeled'}.

    Returns:
        new_params: Updated parameters after applying the masked gradient step.
    """
    # Convert inputs to float arrays
    p = np.asarray(params, dtype=float)
    g = np.asarray(grad, dtype=float)
    m = np.asarray(forget_mask, dtype=float)

    # Binarize the mask: any nonzero value becomes 1.0 (forget), 0 stays retain
    m = (m != 0).astype(float)
    retain_m = 1.0 - m

    if batch_type == "forget":
        # Forget batch: only forget parameters should be updated
        update_mask = m
    elif batch_type == "retain":
        # Retain batch: only retain parameters should be updated
        update_mask = retain_m
    elif batch_type == "unlabeled":
        # Unlabeled batch: all parameters are updated normally
        update_mask = np.ones_like(m)
    else:
        raise ValueError("batch_type must be 'forget', 'retain', or 'unlabeled'")

    # Masked gradient actually used to update the parameters
    masked_grad = g * update_mask

    # Standard SGD update using only the masked gradient
    new_params = p - lr * masked_grad
    return new_params