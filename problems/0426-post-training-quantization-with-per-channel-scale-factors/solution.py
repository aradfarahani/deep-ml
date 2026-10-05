import numpy as np

def per_channel_quantize(weight: np.ndarray, bits: int = 8) -> tuple:
    """
    Perform symmetric per-channel post-training quantization.

    Args:
        weight: Weight matrix of shape (out_channels, in_features)
        bits: Target bit-width for quantization (default: 8)

    Returns:
        Tuple of (quantized_weights, scale_factors, dequantized_weights)
        - quantized_weights: int array of shape (out_channels, in_features)
        - scale_factors: float array of shape (out_channels,)
        - dequantized_weights: float array of shape (out_channels, in_features)
    """
    weight = np.array(weight, dtype=np.float64)
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))

    # Per-channel (per-row) maximum absolute value
    max_abs = np.max(np.abs(weight), axis=1)

    # Scale factors with zero-channel protection
    scale = np.where(max_abs == 0, 1.0, max_abs / qmax)

    # Quantize: scale, round, clip
    quantized = np.clip(np.round(weight / scale[:, np.newaxis]), qmin, qmax).astype(int)

    # Dequantize: multiply back by scale
    dequantized = quantized.astype(np.float64) * scale[:, np.newaxis]

    return quantized, scale, dequantized