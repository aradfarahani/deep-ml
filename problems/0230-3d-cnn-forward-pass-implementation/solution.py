import numpy as np

def conv3d_forward_pass(
    input_volume: np.ndarray,
    kernel: np.ndarray,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Perform 3D convolution forward pass.
    
    Args:
        input_volume: (C, D, H, W)
        kernel: (C, kD, kH, kW)
        stride: (stride_d, stride_h, stride_w)
        padding: (pad_d, pad_h, pad_w)
    
    Returns:
        Output volume of shape (1, D_out, H_out, W_out)
    """
    C, D, H, W = input_volume.shape
    C_k, kD, kH, kW = kernel.shape
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    
    # Apply padding
    if sum(padding) > 0:
        input_padded = np.pad(
            input_volume,
            ((0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            mode='constant'
        )
    else:
        input_padded = input_volume
    
    # Calculate output dimensions
    D_padded, H_padded, W_padded = input_padded.shape[1:]
    D_out = (D_padded - kD) // stride_d + 1
    H_out = (H_padded - kH) // stride_h + 1
    W_out = (W_padded - kW) // stride_w + 1
    
    # Initialize output
    output = np.zeros((1, D_out, H_out, W_out))
    
    # Perform 3D convolution
    for d in range(D_out):
        for h in range(H_out):
            for w in range(W_out):
                d_start = d * stride_d
                h_start = h * stride_h
                w_start = w * stride_w
                
                patch = input_padded[
                    :,
                    d_start:d_start + kD,
                    h_start:h_start + kH,
                    w_start:w_start + kW
                ]
                
                output[0, d, h, w] = np.sum(patch * kernel)
    
    return output