import numpy as np

def fp8_block_quantize(
    tensor: np.ndarray,
    block_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize a tensor to FP8-E4M3 format using block-wise scaling.
    """
    FP8_MAX = 448.0  # Maximum value for E4M3 format
    EPS = 1e-12      # Small constant for numerical stability
    
    # Reshape into blocks
    n_blocks = len(tensor) // block_size
    blocks = tensor.reshape(n_blocks, block_size)
    
    # Compute per-block scales
    # Scale maps max absolute value in block to FP8_MAX
    block_max = np.max(np.abs(blocks), axis=1)
    scales = (block_max + EPS) / FP8_MAX
    
    # Quantize: divide by scale and round
    # scales has shape (n_blocks,), need to broadcast to (n_blocks, block_size)
    quantized_blocks = np.round(blocks / scales[:, np.newaxis])
    
    # Clip to FP8 range
    quantized_blocks = np.clip(quantized_blocks, -FP8_MAX, FP8_MAX)
    
    # Reshape back to original shape
    quantized = quantized_blocks.reshape(-1)
    
    return quantized, scales


def fp8_block_dequantize(
    quantized: np.ndarray,
    scales: np.ndarray,
    block_size: int = 128
) -> np.ndarray:
    """
    Dequantize FP8-E4M3 values back to full precision.
    """
    # Reshape into blocks
    n_blocks = len(quantized) // block_size
    quantized_blocks = quantized.reshape(n_blocks, block_size)
    
    # Dequantize: multiply by scale
    dequantized_blocks = quantized_blocks * scales[:, np.newaxis]
    
    # Reshape back to original shape
    dequantized = dequantized_blocks.reshape(-1)
    
    return dequantized