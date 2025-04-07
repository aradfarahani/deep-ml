import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
    # Compute the average across height and width for each channel
    return np.mean(x, axis=(0, 1))

# Test case
x = np.array([[[1, 2, 3], [4, 5, 6]], 
              [[7, 8, 9], [10, 11, 12]]])

