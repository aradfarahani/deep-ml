import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    n_samples = X_minority.shape[0]
    n_features = X_minority.shape[1]
    k_actual = min(k, n_samples - 1)
    if k_actual == 0 or n_synthetic == 0:
        return np.array([]).reshape(0, n_features)
    synthetic_samples = []
    for _ in range(n_synthetic):
        idx = np.random.randint(0, n_samples)
        sample = X_minority[idx]
        distances = np.sqrt(np.sum((X_minority - sample) ** 2, axis=1))
        neighbor_indices = np.argsort(distances)[1:k_actual + 1]
        nn_idx = neighbor_indices[np.random.randint(0, k_actual)]
        neighbor = X_minority[nn_idx]
        gap = np.random.random()
        synthetic_samples.append(sample + gap * (neighbor - sample))
    return np.array(synthetic_samples)