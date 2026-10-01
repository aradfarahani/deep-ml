import numpy as np

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    N, C, H, W = x.shape
    # Ceil mode formula for output dimensions
    out_h = int(np.ceil((H - kernel_size) / stride)) + 1
    out_w = int(np.ceil((W - kernel_size) / stride)) + 1
    # Use the same dtype as input x (or int) to avoid floating-point output
    pooled = np.zeros((N, C, out_h, out_w), dtype=x.dtype)

    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    h_end = min(h_start + kernel_size, H)  # Clamp to input bounds
                    w_start = j * stride
                    w_end = min(w_start + kernel_size, W)  # Clamp to input bounds
                    window = x[n, c, h_start:h_end, w_start:w_end]
                    pooled[n, c, i, j] = np.max(window)
    
    return pooled