import numpy as np

def compute_ptx_loss(
    rl_loss: float,
    pretrain_logits: np.ndarray,
    pretrain_labels: np.ndarray,
    beta_ptx: float = 0.1
) -> tuple[float, float, float]:
    """
    Compute PTX (Pre-training) Loss for RLHF.
    
    Args:
        rl_loss: Reinforcement learning loss
        pretrain_logits: Shape (batch_size, vocab_size)
        pretrain_labels: Shape (batch_size,)
        beta_ptx: Weight coefficient
    
    Returns:
        (total_loss, ce_loss, weighted_ce_loss)
    """
    # Stable softmax
    exp_logits = np.exp(pretrain_logits - np.max(pretrain_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Cross-entropy
    batch_size = pretrain_labels.shape[0]
    ce_loss = 0.0
    for i in range(batch_size):
        true_label = pretrain_labels[i]
        ce_loss += -np.log(probs[i, true_label] + 1e-10)
    
    ce_loss = ce_loss / batch_size
    
    # Weighted and total
    weighted_ce_loss = beta_ptx * ce_loss
    total_loss = rl_loss + weighted_ce_loss
    
    return total_loss, ce_loss, weighted_ce_loss