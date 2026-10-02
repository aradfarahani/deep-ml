import numpy as np

def rl_budget_loss(
    rewards: np.ndarray,
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    response_lengths: np.ndarray,
    token_budget: int,
    kl_coef: float,
    budget_penalty_coef: float
) -> float:
    """
    Compute the budget-constrained RL loss.
    """
    # Step 1: Compute budget penalties
    # Penalty for exceeding budget: -Î» * max(0, length - budget)
    excess_tokens = np.maximum(0, response_lengths - token_budget)
    budget_penalties = -budget_penalty_coef * excess_tokens
    
    # Step 2: Adjust rewards with budget penalties
    adjusted_rewards = rewards + budget_penalties
    
    # Step 3: Compute baseline (mean adjusted reward per prompt)
    # Shape: (batch_size, 1) for broadcasting
    baseline = np.mean(adjusted_rewards, axis=1, keepdims=True)
    
    # Step 4: Compute advantages
    advantages = adjusted_rewards - baseline
    
    # Step 5: Compute KL terms: Ï * log(Ï_Î¸ / Ï_old) = Ï * (log_Ï_Î¸ - log_Ï_old)
    kl_terms = kl_coef * (log_probs - old_log_probs)
    
    # Step 6: Compute squared loss: (advantage - kl_term)^2
    loss_terms = (advantages - kl_terms) ** 2
    
    # Step 7: Average over all samples
    return float(np.mean(loss_terms))