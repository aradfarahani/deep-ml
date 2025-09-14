import numpy as np

def simulate_clt(num_samples: int, sample_size: int, distribution: str = 'uniform') -> float:
    sample_means = []
    for _ in range(num_samples):
        if distribution == 'uniform':
            data = np.random.uniform(0, 1, sample_size)
        elif distribution == 'exponential':
            data = np.random.exponential(1, sample_size)
        else:
            raise ValueError("Unsupported distribution")
        sample_means.append(np.mean(data))
    return float(np.mean(sample_means))
    