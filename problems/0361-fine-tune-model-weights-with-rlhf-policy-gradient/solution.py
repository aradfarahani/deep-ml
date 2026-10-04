import numpy as np

def rlhf_weight_update(
    weights: np.ndarray,
    rewards: np.ndarray,
    policy_log_probs: np.ndarray,
    ref_log_probs: np.ndarray,
    log_prob_grads: np.ndarray,
    beta: float,
    lr: float
) -> np.ndarray:
    kl = policy_log_probs - ref_log_probs
    adjusted_rewards = rewards - beta * kl
    policy_gradient = np.mean(adjusted_rewards[:, None] * log_prob_grads, axis=0)
    return weights + lr * policy_gradient