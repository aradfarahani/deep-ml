import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    arr = np.asarray(arr, dtype=float)

    if norm_type == 'l1':
        # Entrywise: sum of absolute values
        return float(np.sum(np.abs(arr)))
    elif norm_type == 'l2':
        # Entrywise: square root of the sum of squares
        return float(np.sqrt(np.sum(arr ** 2)))
    elif norm_type == 'linf':
        # Entrywise: largest absolute value
        return float(np.max(np.abs(arr)))
    elif norm_type == 'frobenius':
        # Defined for matrices only -- on a vector it would just be the L2 norm
        if arr.ndim != 2:
            raise ValueError(
                f"The Frobenius norm is defined for matrices, got a {arr.ndim}D array"
            )
        return float(np.linalg.norm(arr, 'fro'))
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")
