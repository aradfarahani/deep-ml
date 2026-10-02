import numpy as np

def qlora_forward(
    x: list[list[float]],
    quantized_W: list[list[int]],
    scale: float,
    zero_point: float,
    A: list[list[float]],
    B: list[list[float]],
    alpha: float = 1.0
) -> list[list[float]]:
    """
    QLoRA forward pass with 4-bit quantized frozen weights.
    
    Args:
        x: Input (batch_size x in_features)
        quantized_W: 4-bit quantized frozen weights (in_features x out_features)
        scale: Quantization scale factor
        zero_point: Quantization zero point
        A: LoRA A matrix (rank x out_features)
        B: LoRA B matrix (in_features x rank)
        alpha: Scaling factor
        
    Returns:
        Output (batch_size x out_features)
    """
    x = np.array(x)
    quantized_W = np.array(quantized_W)
    A = np.array(A)
    B = np.array(B)
    
    # Dequantize frozen weights: W = quantized_W * scale + zero_point
    W = quantized_W.astype(np.float64) * scale + zero_point
    
    rank = A.shape[0]
    
    # Forward pass: frozen path + LoRA path
    output = x @ W + (alpha / rank) * (x @ B @ A)
    
    return output.tolist()