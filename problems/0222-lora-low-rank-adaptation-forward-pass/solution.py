import numpy as np

def lora_forward(
    x: list[list[float]],
    W: list[list[float]],
    A: list[list[float]],
    B: list[list[float]],
    alpha: float = 1.0
) -> list[list[float]]:
    """
    Compute LoRA forward pass: output = x @ W + (alpha/r) * x @ B @ A
    
    Args:
        x: Input (batch_size x in_features)
        W: Frozen weights (in_features x out_features)
        A: LoRA A matrix (rank x out_features)
        B: LoRA B matrix (in_features x rank)
        alpha: Scaling factor
        
    Returns:
        Output (batch_size x out_features)
    """
    x = np.array(x)
    W = np.array(W)
    A = np.array(A)
    B = np.array(B)
    
    rank = A.shape[0]
    
    # Frozen path + LoRA path with scaling
    output = x @ W + (alpha / rank) * (x @ B @ A)
    
    return output.tolist()