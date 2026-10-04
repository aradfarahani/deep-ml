import numpy as np

def dpo_loss(log_probs_chosen_policy: list, log_probs_rejected_policy: list,
            log_probs_chosen_ref: list, log_probs_rejected_ref: list,
            beta: float) -> dict:
    """
    Compute the Direct Preference Optimization (DPO) loss.
    
    Args:
        log_probs_chosen_policy: Log-probs of chosen responses under policy
        log_probs_rejected_policy: Log-probs of rejected responses under policy
        log_probs_chosen_ref: Log-probs of chosen responses under reference model
        log_probs_rejected_ref: Log-probs of rejected responses under reference model
        beta: Temperature parameter for KL constraint strength
    
    Returns:
        Dictionary with 'loss', 'chosen_rewards', and 'rejected_rewards'
    """
    log_probs_chosen_policy = np.array(log_probs_chosen_policy, dtype=float)
    log_probs_rejected_policy = np.array(log_probs_rejected_policy, dtype=float)
    log_probs_chosen_ref = np.array(log_probs_chosen_ref, dtype=float)
    log_probs_rejected_ref = np.array(log_probs_rejected_ref, dtype=float)
    
    # Compute log-ratios between policy and reference
    chosen_log_ratios = log_probs_chosen_policy - log_probs_chosen_ref
    rejected_log_ratios = log_probs_rejected_policy - log_probs_rejected_ref
    
    # Compute logits for the DPO loss
    logits = beta * (chosen_log_ratios - rejected_log_ratios)
    
    # Compute loss: -log(sigmoid(logits))
    # Using numerically stable form: log(1 + exp(-logits))
    losses = np.log(1.0 + np.exp(-logits))
    
    # Average loss over the batch
    loss = float(np.mean(losses))
    
    # Implicit rewards
    chosen_rewards = beta * chosen_log_ratios
    rejected_rewards = beta * rejected_log_ratios
    
    return {
        'loss': round(loss, 4),
        'chosen_rewards': [round(float(r), 4) for r in chosen_rewards],
        'rejected_rewards': [round(float(r), 4) for r in rejected_rewards]
    }