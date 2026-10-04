import numpy as np

def moe_load_balancing_loss(gate_logits, num_experts, alpha=0.01):
    """
    Compute the load balancing auxiliary loss for a Mixture of Experts layer.
    
    Args:
        gate_logits: numpy array of shape (num_tokens, num_experts), raw gating scores
        num_experts: int, number of experts
        alpha: float, scaling coefficient for the loss
    
    Returns:
        float: load balancing loss rounded to 4 decimal places
    """
    gate_logits = np.array(gate_logits, dtype=np.float64)
    num_tokens = gate_logits.shape[0]
    
    # Compute softmax routing probabilities (numerically stable)
    shifted = gate_logits - np.max(gate_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Hard routing: assign each token to the expert with highest probability
    assignments = np.argmax(probs, axis=-1)
    
    # f_i: fraction of tokens dispatched to each expert
    f = np.zeros(num_experts, dtype=np.float64)
    for i in range(num_experts):
        f[i] = np.sum(assignments == i) / num_tokens
    
    # P_i: mean routing probability for each expert across all tokens
    P = np.mean(probs, axis=0)
    
    # Load balancing loss: alpha * N * sum(f_i * P_i)
    loss = alpha * num_experts * np.sum(f * P)
    
    return round(float(loss), 4)