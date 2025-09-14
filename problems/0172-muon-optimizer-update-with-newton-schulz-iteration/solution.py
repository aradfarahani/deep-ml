import numpy as np

def newtonschulz5(G: np.ndarray, steps=5, eps=1e-7) -> np.ndarray:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(np.float32)
    orig_shape = X.shape
    if X.shape[0] > X.shape[1]:
        X = X.T
    norm = np.sqrt(np.sum(X**2)) + eps
    X = X / norm
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if orig_shape[0] > orig_shape[1]:
        X = X.T
    return X

def muon_update(theta: np.ndarray, grad: np.ndarray, B_prev: np.ndarray, mu: float, lr: float) -> tuple:
    # Step 1: Update momentum
    B = mu * B_prev + grad
    # Step 2: Precondition with Newton-Schulz5
    O = newtonschulz5(B)
    # Step 3: Parameter update
    theta_new = theta - lr * O
    return theta_new, B, O