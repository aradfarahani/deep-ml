import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_gan(mean_real: float, std_real: float, latent_dim: int = 1, hidden_dim: int = 16, learning_rate: float = 0.001, epochs: int = 5000, batch_size: int = 128, seed: int = 42):
    np.random.seed(seed)
    data_dim = 1

    # Initialize generator weights
    w1_g = np.random.normal(0, 0.01, (latent_dim, hidden_dim))
    b1_g = np.zeros(hidden_dim)
    w2_g = np.random.normal(0, 0.01, (hidden_dim, data_dim))
    b2_g = np.zeros(data_dim)

    # Initialize discriminator weights
    w1_d = np.random.normal(0, 0.01, (data_dim, hidden_dim))
    b1_d = np.zeros(hidden_dim)
    w2_d = np.random.normal(0, 0.01, (hidden_dim, 1))
    b2_d = np.zeros(1)

    def disc_forward(x):
        h1 = np.dot(x, w1_d) + b1_d
        a1 = relu(h1)
        logit = np.dot(a1, w2_d) + b2_d
        p = sigmoid(logit)
        return p, logit, a1, h1

    def gen_forward(z):
        h1 = np.dot(z, w1