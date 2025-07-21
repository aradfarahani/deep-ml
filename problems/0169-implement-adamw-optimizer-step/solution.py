import numpy as np

def adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay):
    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * (g ** 2)
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    w = w - lr * weight_decay * w  # decoupled weight decay
    w_new = w - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return w_new, m_new, v_new
    