import numpy as np

def sparse_window_attention(Q, K, V, window_size, scale_factor=None):
    """
    Computes sparse attention with a sliding window mask to efficiently handle longer context lengths.
    This implementation uses a loop over the sequence to compute attention only within the specified window,
    reducing memory usage compared to dense attention.

    Args:
        Q (np.ndarray): Query matrix of shape (seq_len, d_k)
        K (np.ndarray): Key matrix of shape (seq_len, d_k)
        V (np.ndarray): Value matrix of shape (seq_len, d_v)
        window_size (int): The radius of the attention window (attends to window_size positions on each side).
        scale_factor (float, optional): Scaling factor for the dot product. If None, uses sqrt(d_k).

    Returns:
        np.ndarray: Attention output of shape (seq_len, d_v)
    """
    seq_len = Q.shape[0]
    d_k = Q.shape[1]
    if scale_factor is None:
        scale_factor = np.sqrt(d_k).astype(float)
    output = np.zeros((