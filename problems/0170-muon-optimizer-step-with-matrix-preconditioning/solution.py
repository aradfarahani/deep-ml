import numpy as np
def newton_schulz5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(np.float32)
    X /= np.linalg.norm(X, 'fro') + eps
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if transposed:
        X = X.T
    return X

def muon_step(theta, B, grad, eta, mu, ns_steps=5, eps=1e-7):
    B_new = mu * B + grad
    O = newton_schulz5(B_new, steps=ns_steps)
    scale = np.sqrt(np.prod(theta.shape)) / (np.linalg.norm(B_new, 'fro') + eps)
    theta_new = theta - eta * scale * O
    return theta_new, B_new