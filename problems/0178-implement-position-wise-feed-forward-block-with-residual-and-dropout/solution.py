import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float=0.1, seed: int=42) -> list[float]:
    np.random.seed(seed)
    x = np.array(x)
    W1, b1, W2, b2 = np.array(W1), np.array(b1), np.array(W2), np.array(b2)
    # First linear + ReLU
    hidden = np.maximum(0, W1.dot(x) + b1)
    # Second linear
    out = W2.dot(hidden) + b2
    # Dropout (applied BEFORE residual, per Transformer standard)
    mask = (np.random.rand(*out.shape) > dropout_p).astype(float)
    out = out * mask / (1 - dropout_p)
    # Residual connection
    out = out + x
    return [round(v, 4) for v in out.tolist()]