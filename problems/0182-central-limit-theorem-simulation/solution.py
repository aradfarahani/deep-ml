import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    np.random.seed(seed)
    if distribution == 'uniform':
        data = np.random.uniform(0, 1, size=(runs, n))
        mu, sigma = 0.5, np.sqrt(1/12)
    elif distribution == 'exponential':
        data = np.random.exponential(1.0, size=(runs, n))
        mu, sigma = 1.0, 1.0
    elif distribution == 'bernoulli':
        p = 0.3
        data = (np.random.rand(runs, n) < p).astype(float)
        mu, sigma = p, np.sqrt(p*(1-p))
    else:
        raise ValueError('Unsupported distribution')

    xbar = data.mean(axis=1)
    z = (xbar - mu) / (sigma / np.sqrt(n))
    return {"mean": float(z.mean()), "std": float(z.std())}