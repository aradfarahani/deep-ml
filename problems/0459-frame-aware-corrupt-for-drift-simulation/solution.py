import numpy as np

def frame_aware_corrupt(
    frames: np.ndarray,
    gaussian_prob: float,
    gaussian_std: float,
    color_shift_prob: float,
    color_shift_range: float,
    blur_prob: float,
    blur_kernel_size: int,
    rng: np.random.Generator
) -> np.ndarray:
    frames = frames.astype(float)
    corrupted = frames.copy()
    T = frames.shape[0]

    for t in range(T):
        # Gaussian noise
        if rng.random() < gaussian_prob:
            corrupted[t] += rng.normal(0, gaussian_std, corrupted[t].shape)

        # Per-channel color shift
        if rng.random() < color_shift_prob:
            shift = rng.uniform(-color_shift_range, color_shift_range, (corrupted[t].shape[-1],))
            corrupted[t] += shift

        # Mean blur with edge padding
        if rng.random() < blur_prob:
            k = blur_kernel_size
            kernel = np.ones((k, k)) / (k * k)
            pad = k // 2
            for c in range(corrupted[t].shape[-1]):
                ch = corrupted[t, :, :, c]
                padded = np.pad(ch, pad, mode='edge')
                blurred = np.zeros_like(ch)
                for i in range(ch.shape[0]):
                    for j in range(ch.shape[1]):
                        blurred[i, j] = np.sum(padded[i:i+k, j:j+k] * kernel)
                corrupted[t, :, :, c] = blurred

    return corrupted